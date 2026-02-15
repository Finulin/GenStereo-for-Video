import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="diffusers")

import os
from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import argparse
import subprocess
from tqdm import tqdm
import gc
import platform

from genstereo import GenStereo, AdaptiveFusionLayer

# Real-ESRGAN setup
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

# Detect macOS
IS_MACOS = platform.system() == 'Darwin'

def get_device(use_mps=False):
    """Get device based on user preference and system."""
    if use_mps and IS_MACOS and torch.backends.mps.is_available():
        print("[INFO] Using MPS (Metal) - if crashes occur, run without --mps flag")
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        if IS_MACOS:
            print("[INFO] Using CPU mode (stable but slower)")
        else:
            print("[INFO] Using CPU mode")
        return 'cpu'

def get_sd_version(fast_mode=False):
    """Get SD version - v1.5 is faster (512x512), v2.1 is better quality (768x768)."""
    if fast_mode:
        return "v1.5", 512, 'genstereo-v1.5'
    else:
        return "v2.1", 768, 'genstereo-v2.1'

checkpoint_dir = 'checkpoints'

# Global models (lazy loading)
_genstereo_nvs = None
_fusion_model = None
_current_sd_version = None
_current_device = None

def load_models(sd_version, checkpoint_name, device):
    """Load models lazily."""
    global _genstereo_nvs, _fusion_model, _current_sd_version, _current_device

    if _genstereo_nvs is not None and _current_sd_version == sd_version and _current_device == device:
        return _genstereo_nvs, _fusion_model

    print(f"Loading GenStereo ({sd_version}) on {device}...")

    genstereo_cfg = dict(
        pretrained_model_path=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        half_precision_weights=False,  # Safer for MPS/CPU
    )

    _genstereo_nvs = GenStereo(cfg=genstereo_cfg, device=device, sd_version=sd_version)

    _fusion_model = AdaptiveFusionLayer()
    fusion_checkpoint = join(checkpoint_dir, checkpoint_name, 'fusion_layer.pth')
    _fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location=device))
    _fusion_model = _fusion_model.to(device).eval()

    _current_sd_version = sd_version
    _current_device = device

    print("Models loaded.")
    return _genstereo_nvs, _fusion_model

# Real-ESRGAN upscaler (lazy initialization)
_upsampler = None
_upsampler_scale = None

def get_upsampler(scale=4, device='cpu'):
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

    # Use CPU for Real-ESRGAN to save GPU memory
    _upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device='cpu'
    )
    _upsampler_scale = scale

    return _upsampler

def upscale_frame(img, scale=4, device='cpu'):
    """Upscale a single frame using Real-ESRGAN."""
    upsampler = get_upsampler(scale, device)
    img_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output, _ = upsampler.enhance(img_np, outscale=scale)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

class FFmpegVideoWriter:
    """Video writer using FFmpeg with H.264 codec and BT.709 colorspace."""

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
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-r', str(self.fps),
            '-i', 'pipe:',
            '-c:v', 'libx264',
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-loglevel', 'error',
            self.output_path
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def write(self, frame):
        """Write RGB frame (numpy array) to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame size mismatch. Expected {self.width}x{self.height}")
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
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
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def clear_memory(device='cpu'):
    """Clear memory."""
    gc.collect()
    if device == 'mps':
        try:
            torch.mps.empty_cache()
        except:
            pass
    elif device == 'cuda':
        torch.cuda.empty_cache()

def process_frame(image_pil, depth_tensor, convergence, image_size, device, genstereo_nvs, fusion_model):
    """Process a single frame and return left and right images as numpy arrays (RGB)."""

    disparity = normalize_disp(depth_tensor) * convergence * image_size

    with torch.inference_mode():
        renders = genstereo_nvs(src_image=image_pil, src_disparity=disparity, ratio=None)
        warped = (renders['warped'] + 1) / 2
        mask = morphological_opening(renders['mask']).to(device)
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())

    left_np = np.array(image_pil)
    right_np = np.array(to_pil_image(fusion_image[0].cpu()))

    del renders, warped, mask, fusion_image, disparity

    return left_np, right_np

def process_video(video_path, depth_video_path, output_dir, convergence=0.02, upscale=2,
                  save_all=False, crf=23, preset='medium', clear_every=10,
                  fast_mode=False, use_mps=False):
    """Process video frame by frame and save output videos."""

    if not check_ffmpeg():
        print("[ERROR] FFmpeg not found!")
        print("  Windows: choco install ffmpeg  OR  download from https://ffmpeg.org")
        print("  macOS:   brew install ffmpeg")
        return

    # Get SD version and settings
    sd_version, image_size, checkpoint_name = get_sd_version(fast_mode)
    device = get_device(use_mps)

    # Load models
    genstereo_nvs, fusion_model = load_models(sd_version, checkpoint_name, device)

    base_name = splitext(basename(video_path))[0]
    output_path = join(output_dir, base_name)
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    depth_cap = cv2.VideoCapture(depth_video_path)
    if not depth_cap.isOpened():
        raise RuntimeError(f"Failed to open depth video: {depth_video_path}")

    depth_total = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_size = calculate_output_size((width, height), image_size)
    out_w, out_h = output_size

    if upscale > 0:
        final_w = out_w * upscale
        final_h = out_h * upscale
    else:
        final_w, final_h = out_w, out_h

    print(f"")
    print(f"SD Version:    {sd_version} ({image_size}x{image_size})")
    print(f"Device:        {device}")
    print(f"Input video:   {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"Depth video:   {depth_total} frames")
    print(f"Output size:   {final_w}x{final_h}")
    print(f"CRF:           {crf}, Preset: {preset}")
    print(f"Clear memory:  Every {clear_every} frames")
    print(f"")

    left_writer = FFmpegVideoWriter(join(output_path, 'left.mp4'), final_w, final_h, fps, crf, preset)
    right_writer = FFmpegVideoWriter(join(output_path, 'right.mp4'), final_w, final_h, fps, crf, preset)

    if save_all:
        disp_writer = FFmpegVideoWriter(join(output_path, 'disp.mp4'), final_w, final_h, fps, crf, preset)

    frame_idx = 0
    pbar = tqdm(total=total_frames, unit='frame', desc='Processing')

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            ret_depth, depth_frame_bgr = depth_cap.read()
            if not ret_depth:
                print(f"\n[WARNING] Depth video ended at frame {frame_idx}")
                break

            # Process depth
            depth_gray = cv2.cvtColor(depth_frame_bgr, cv2.COLOR_BGR2GRAY)
            depth_pil = Image.fromarray(depth_gray).resize((image_size, image_size), Image.BILINEAR)
            depth_tensor = to_tensor(depth_pil).unsqueeze(0).float().to(device)

            # Process frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((image_size, image_size), Image.BILINEAR)

            # Main processing
            left_np, right_np = process_frame(
                frame_pil, depth_tensor, convergence, image_size, device,
                genstereo_nvs, fusion_model
            )

            # Resize to output size
            left_np = cv2.resize(left_np, (out_w, out_h))
            right_np = cv2.resize(right_np, (out_w, out_h))

            # Upscale if requested
            if upscale > 0:
                left_np = upscale_frame(left_np, upscale, device)
                right_np = upscale_frame(right_np, upscale, device)

            # Write frames
            left_writer.write(left_np)
            right_writer.write(right_np)

            if save_all:
                depth_vis = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
                depth_vis = cv2.resize(depth_vis, (out_w, out_h))
                depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
                if upscale > 0:
                    depth_vis_rgb = upscale_frame(depth_vis_rgb, upscale, device)
                disp_writer.write(depth_vis_rgb)

            # Clean up
            del depth_tensor, frame_pil, depth_pil, left_np, right_np

            frame_idx += 1
            pbar.update(1)

            # Periodic memory cleanup
            if frame_idx % clear_every == 0:
                clear_memory(device)

        pbar.close()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == 'mps':
            print(f"\n[ERROR] MPS out of memory!")
            print("  Try: Run without --mps flag to use CPU (slower but stable)")
            print("  Or:  Use --fast for smaller model (512x512 instead of 768x768)")
        raise

    finally:
        cap.release()
        depth_cap.release()
        left_writer.close()
        right_writer.close()
        if save_all:
            disp_writer.close()
        clear_memory(device)

    print(f"\nDone! Processed {frame_idx} frames")
    print(f"Output directory: {output_path}/")
    print(f"  - left.mp4")
    print(f"  - right.mp4")
    if save_all:
        print(f"  - disp.mp4")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stereo video from input video with depth map")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("depth_video_path", help="Path to depth video (grayscale)")
    parser.add_argument("--output", default="./vis", help="Output directory")
    parser.add_argument("--convergence", type=float, default=0.02, help="Convergence factor for stereo depth")
    parser.add_argument("--save_all", action='store_true', help="Save disp.mp4 in addition to left.mp4 and right.mp4")
    parser.add_argument("--upscale", type=int, choices=[0, 2, 4], default=2, help="Upscale output videos by factor (0=no upscaling, 2=2x, 4=4x)")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality (0-51, lower=better, 18-28 recommended)")
    parser.add_argument("--preset", default='medium', help="Encoding speed (ultrafast, fast, medium, slow)")
    parser.add_argument("--clear_every", type=int, default=10, help="Clear memory every N frames")
    parser.add_argument("--fast", action='store_true', help="Use SD v1.5 (512x512) - 3-4x faster, slightly lower quality")
    parser.add_argument("--mps", action='store_true', help="Try to use MPS (Metal) on macOS - faster but may crash if out of memory")

    args = parser.parse_args()

    print(f"")
    if args.fast:
        print(f"[FAST MODE] Using SD v1.5 (512x512) - significantly faster")
    else:
        print(f"[QUALITY MODE] Using SD v2.1 (768x768) - slower but better quality")
        print(f"               Use --fast for 3-4x speedup on CPU")

    if args.mps and IS_MACOS:
        print(f"[MPS] Attempting to use Metal GPU - may crash if insufficient memory")

    if args.upscale > 0:
        if not REALESRGAN_AVAILABLE:
            print("")
            print("[WARNING] Real-ESRGAN not installed!")
            print("  Install with: pip install basicsr realesrgan")
            print("  Continuing without upscaling...")
            args.upscale = 0
        else:
            print(f"Upscaling: {args.upscale}x (Real-ESRGAN on CPU)")

    process_video(
        args.video_path,
        args.depth_video_path,
        args.output,
        convergence=args.convergence,
        upscale=args.upscale,
        save_all=args.save_all,
        crf=args.crf,
        preset=args.preset,
        clear_every=args.clear_every,
        fast_mode=args.fast,
        use_mps=args.mps
    )
