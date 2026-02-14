import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="diffusers")

from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import os
from PIL import Image
import argparse
import subprocess
from tqdm import tqdm
import gc

from genstereo import GenStereo, AdaptiveFusionLayer

# Real-ESRGAN setup
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

SD_VERSION = "v2.1"
if SD_VERSION == "v1.5":
    IMAGE_SIZE = 512
    CHECKPOINT_NAME = 'genstereo-v1.5'
elif SD_VERSION == "v2.1":
    IMAGE_SIZE = 768
    CHECKPOINT_NAME = 'genstereo-v2.1'
else:
    raise ValueError(f"Unknown SD version: {SD_VERSION}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

checkpoint_dir = 'checkpoints'

genstereo_cfg = dict(
    pretrained_model_path=checkpoint_dir,
    checkpoint_name=CHECKPOINT_NAME,
    half_precision_weights=True if 'cuda' in DEVICE else False,
)
genstereo_nvs = GenStereo(cfg=genstereo_cfg, device=DEVICE, sd_version=SD_VERSION)

fusion_model = AdaptiveFusionLayer()
fusion_checkpoint = join(checkpoint_dir, CHECKPOINT_NAME, 'fusion_layer.pth')
fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location=DEVICE))
fusion_model = fusion_model.to(DEVICE).eval()

# Real-ESRGAN upscaler (lazy initialization)
_upsampler = None
_upsampler_scale = None

def get_upsampler(scale=4):
    """Initialize and return Real-ESRGAN upsampler."""
    global _upsampler, _upsampler_scale
    
    if _upsampler is not None and _upsampler_scale == scale:
        return _upsampler
    
    if not REALESRGAN_AVAILABLE:
        raise ImportError("Real-ESRGAN not installed. Install with: pip install basicsr realesrgan")
    
    esrgan_dir = 'checkpoints/esrgan'
    os.makedirs(esrgan_dir, exist_ok=True)
    
    if scale == 2:
        model_name = 'RealESRGAN_x2plus'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        model_name = 'RealESRGAN_x4plus'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model_path = join(esrgan_dir, f'{model_name}.pth')
    
    if not os.path.exists(model_path):
        print(f"Downloading Real-ESRGAN model: {model_name}...")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded to: {model_path}")
    
    upscaler_device = DEVICE
    if DEVICE == 'mps':
        upscaler_device = 'cpu'
    
    _upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=512,          # Use tiling to reduce VRAM
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=upscaler_device
    )
    _upsampler_scale = scale
    
    return _upsampler

def upscale_frame(img, scale=4):
    """Upscale a single frame using Real-ESRGAN."""
    upsampler = get_upsampler(scale)
    
    if isinstance(img, Image.Image):
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img_np = img
    
    output, _ = upsampler.enhance(img_np, outscale=scale)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def get_cuda_memory_info():
    """Get CUDA memory usage info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"VRAM: {allocated:.2f}GB used, {reserved:.2f}GB reserved"
    return ""

class FFmpegVideoWriter:
    """Video writer using FFmpeg with H.264 codec."""
    
    def __init__(self, output_path, width, height, fps, crf=23, preset='medium'):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.process = None
        self._start_process()
    
    def _start_process(self):
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', 'pipe:',
            '-c:v', 'libx264',
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            self.output_path
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer
        )
    
    def write(self, frame):
        """Write RGB frame (numpy array) to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame size mismatch. Expected {self.width}x{self.height}")
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()  # Force flush after each frame
        except BrokenPipeError:
            stderr = self.process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg error: {stderr}")
    
    def close(self):
        """Finalize video."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                print("[WARNING] FFmpeg process killed (timeout)")
            except Exception as e:
                print(f"[WARNING] FFmpeg close error: {e}")

def calculate_output_size(original_size: tuple, max_size: int) -> tuple:
    """Calculate output size keeping aspect ratio."""
    W, H = original_size
    
    if W >= H:
        new_width = max_size
        new_height = int(H * (max_size / W))
    else:
        new_height = max_size
        new_width = int(W * (max_size / H))
    
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)
    
    return (new_width, new_height)

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def process_frame(image_pil, depth_tensor, convergence=0.02):
    """Process a single frame and return left and right images as numpy arrays (RGB)."""
    
    # Prepare disparity
    disparity = normalize_disp(depth_tensor) * convergence * IMAGE_SIZE
    
    # Generate novel view
    renders = genstereo_nvs(src_image=image_pil, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    
    with torch.inference_mode():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
    
    # Convert to numpy RGB
    left_np = np.array(image_pil)
    right_np = np.array(to_pil_image(fusion_image[0]))
    
    return left_np, right_np

def process_video(video_path, depth_video_path, output_dir, convergence=0.02, upscale=2, save_all=False, crf=23, preset='medium', clear_every=10):
    """Process video frame by frame and save output videos.
    
    Args:
        clear_every: Clear memory every N frames to prevent memory buildup
    """
    
    if not check_ffmpeg():
        print("[ERROR] FFmpeg not found!")
        print("  Windows: choco install ffmpeg  OR  download from https://ffmpeg.org")
        print("  macOS:   brew install ffmpeg")
        return
    
    base_name = splitext(basename(video_path))[0]
    output_path = join(output_dir, base_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Open depth video
    depth_cap = cv2.VideoCapture(depth_video_path)
    if not depth_cap.isOpened():
        raise RuntimeError(f"Failed to open depth video: {depth_video_path}")
    
    depth_total = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output size
    output_size = calculate_output_size((width, height), IMAGE_SIZE)
    out_w, out_h = output_size
    
    # Apply upscale factor
    if upscale > 0:
        final_w = out_w * upscale
        final_h = out_h * upscale
    else:
        final_w, final_h = out_w, out_h
    
    print(f"Input video:   {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"Depth video:   {depth_total} frames")
    print(f"Output size:   {final_w}x{final_h}")
    print(f"CRF:           {crf}, Preset: {preset}")
    print(f"Clear memory:  Every {clear_every} frames")
    print("")
    
    # Initialize video writers
    left_writer = FFmpegVideoWriter(join(output_path, 'left.mp4'), final_w, final_h, fps, crf, preset)
    right_writer = FFmpegVideoWriter(join(output_path, 'right.mp4'), final_w, final_h, fps, crf, preset)
    
    if save_all:
        disp_writer = FFmpegVideoWriter(join(output_path, 'disp.mp4'), final_w, final_h, fps, crf, preset)
        warped_writer = FFmpegVideoWriter(join(output_path, 'warped.mp4'), final_w, final_h, fps, crf, preset)
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, unit='frame', desc='Processing')
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # Read depth frame
            ret_depth, depth_frame_bgr = depth_cap.read()
            if not ret_depth:
                print(f"\n[WARNING] Depth video ended at frame {frame_idx}")
                break
            
            # Convert depth frame to grayscale and resize
            depth_gray = cv2.cvtColor(depth_frame_bgr, cv2.COLOR_BGR2GRAY)
            depth_pil = Image.fromarray(depth_gray).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            depth_tensor = to_tensor(depth_pil).unsqueeze(0).float().to(DEVICE)
            
            # Convert frame to RGB and resize to square for model
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            
            # Process frame
            left_np, right_np = process_frame(frame_pil, depth_tensor, convergence)
            
            # Resize to output size
            left_np = cv2.resize(left_np, (out_w, out_h))
            right_np = cv2.resize(right_np, (out_w, out_h))
            
            # Upscale if requested
            if upscale > 0:
                left_np = upscale_frame(left_np, upscale)
                right_np = upscale_frame(right_np, upscale)
            
            # Write frames
            left_writer.write(left_np)
            right_writer.write(right_np)
            
            if save_all:
                depth_vis = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
                depth_vis = cv2.resize(depth_vis, (out_w, out_h))
                if upscale > 0:
                    depth_vis = upscale_frame(depth_vis, upscale)
                    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
                disp_writer.write(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
                
                warped_np = cv2.resize(np.array(to_pil_image((renders['warped'][0] + 1) / 2)), (out_w, out_h))
                if upscale > 0:
                    warped_np = upscale_frame(warped_np, upscale)
                warped_writer.write(warped_np)
            
            frame_idx += 1
            pbar.update(1)
            
            # Periodic memory cleanup
            if frame_idx % clear_every == 0:
                clear_memory()
                if torch.cuda.is_available() and frame_idx % 50 == 0:
                    mem_info = get_cuda_memory_info()
                    pbar.set_postfix_str(mem_info)
        
        pbar.close()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        depth_cap.release()
        left_writer.close()
        right_writer.close()
        if save_all:
            disp_writer.close()
            warped_writer.close()
        clear_memory()
    
    print(f"\nDone! Processed {frame_idx} frames")
    print(f"Output directory: {output_path}/")
    print(f"  - left.mp4")
    print(f"  - right.mp4")
    if save_all:
        print(f"  - disp.mp4")
        print(f"  - warped.mp4")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stereo video from input video with depth map")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("depth_video_path", help="Path to depth video (grayscale)")
    parser.add_argument("--output", default="./vis", help="Output directory")
    parser.add_argument("--convergence", type=float, default=0.02, help="Convergence factor for stereo depth")
    parser.add_argument("--save_all", action='store_true', help="Save all outputs (disp.mp4, warped.mp4) in addition to left.mp4 and right.mp4")
    parser.add_argument("--upscale", type=int, choices=[0, 2, 4], default=2, help="Upscale output videos by factor (0=no upscaling, 2=2x, 4=4x)")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality (0-51, lower=better, 18-28 recommended)")
    parser.add_argument("--preset", default='medium', help="Encoding speed (ultrafast, fast, medium, slow)")
    parser.add_argument("--clear_every", type=int, default=10, help="Clear memory every N frames")
    
    args = parser.parse_args()
    
    print(f"SD Version: {SD_VERSION}")
    print(f"Device: {DEVICE}")
    if args.upscale > 0:
        if not REALESRGAN_AVAILABLE:
            print("")
            print("[WARNING] Real-ESRGAN not installed!")
            print("  Install with: pip install basicsr realesrgan")
            print("  Continuing without upscaling...")
            args.upscale = 0
        else:
            print(f"Upscaling: {args.upscale}x (Real-ESRGAN)")
    print("")
    
    process_video(
        args.video_path,
        args.depth_video_path,
        args.output,
        convergence=args.convergence,
        upscale=args.upscale,
        save_all=args.save_all,
        crf=args.crf,
        preset=args.preset,
        clear_every=args.clear_every
    )