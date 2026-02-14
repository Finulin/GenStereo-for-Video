import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="diffusers")

from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import ssl
import os
from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import argparse
from tqdm import tqdm
import subprocess
import shutil

from genstereo import GenStereo, AdaptiveFusionLayer

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

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder = 'vitl'
encoder_size_map = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}

if encoder not in encoder_size_map:
    raise ValueError(f"Unsupported encoder: {encoder}. Supported: {list(encoder_size_map.keys())}")

dam2 = DepthAnythingV2(**model_configs[encoder])
size_name = encoder_size_map[encoder]
dam2_path = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{size_name}/resolve/main/depth_anything_v2_{encoder}.pth"

checkpoint_dir = 'checkpoints'
dam2_checkpoint = f'{checkpoint_dir}/depth_anything_v2_{encoder}.pth'
os.makedirs(checkpoint_dir, exist_ok=True)

if not os.path.exists(dam2_checkpoint):
    print(f"Downloading DAM2 model from {dam2_path}")
    os.system(f"wget {dam2_path} -O {dam2_checkpoint}")

dam2.load_state_dict(torch.load(dam2_checkpoint, map_location='cpu'))
dam2 = dam2.to(DEVICE).eval()

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

def crop(img: Image) -> Image:
    W, H = img.size
    if W < H:
        crop_size = W
        top = (H - crop_size) // 2
        bottom = top + crop_size
        left, right = 0, W
    else:
        crop_size = H
        left = (W - crop_size) // 2
        right = left + crop_size
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))

def infer_depth_dam2(image_np: np.ndarray) -> torch.Tensor:
    """Infer depth from numpy BGR image array."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_pil = crop(image_pil).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    depth_dam2 = dam2.infer_image(image_bgr)
    return torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

def normalize_disp(disp):
    minv = disp.min()
    maxv = disp.max()
    denom = maxv - minv
    if denom < 1e-6:
        return torch.zeros_like(disp)
    return (disp - minv) / denom

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def process_frame(image_pil: Image, depth: torch.Tensor, scale_factor=0.15):
    """Process a single frame and return output images as numpy arrays (BGR).
    
    Returns:
        dict with keys: 'left', 'warped', 'generated_right', 'disp'
        All values are numpy arrays in BGR format
    """
    disparity = normalize_disp(depth) * scale_factor * IMAGE_SIZE
    
    renders = genstereo_nvs(src_image=image_pil, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    
    with torch.no_grad():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
    
    left_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    warped_pil = to_pil_image(warped[0])
    warped_np = cv2.cvtColor(np.array(warped_pil), cv2.COLOR_RGB2BGR)
    
    fusion_image_pil = to_pil_image(fusion_image[0])
    fusion_np = cv2.cvtColor(np.array(fusion_image_pil), cv2.COLOR_RGB2BGR)
    
    disp_vis = cv2.applyColorMap(
        (normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )
    
    return {
        'left': left_np,
        'warped': warped_np,
        'generated_right': fusion_np,
        'disp': disp_vis
    }

def check_ffmpeg_available():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

class FFmpegVideoWriter:
    """Wrapper for writing video using FFmpeg with H.264 codec."""
    
    def __init__(self, output_path, width, height, fps, crf=23, preset='medium'):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.crf = crf
        self.preset = preset
        self._start_process()
    
    def _start_process(self):
        """Start FFmpeg process."""
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', 'pipe:',
            '-c:v', 'libx264',
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',
            self.output_path
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    
    def write(self, frame):
        """Write a BGR frame (numpy array) to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame size mismatch. Expected {self.width}x{self.height}, got {frame.shape[1]}x{frame.shape[0]}")
        
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"FFmpeg process error: {e}")
    
    def release(self):
        """Finalize video writing."""
        if self.process:
            try:
                self.process.stdin.close()
                stdout, stderr = self.process.communicate(timeout=300)
                if self.process.returncode != 0:
                    print(f"Warning: FFmpeg returned code {self.process.returncode}")
                    if stderr:
                        print(f"FFmpeg stderr: {stderr.decode()[-200:]}")
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise RuntimeError("FFmpeg encoding timeout")
            except Exception as e:
                print(f"Warning during FFmpeg finalization: {e}")
            finally:
                self.process = None

def process_video(video_path, depth_video_path, output_dir, scale_factor=0.15, skip_frames=1, use_ffmpeg=True, crf=23, preset='medium'):
    """Process video frame by frame and save output videos.
    
    Args:
        video_path: Path to input video
        depth_video_path: Path to depth video or None for DAM2 inference
        output_dir: Output directory
        scale_factor: Disparity scale factor
        skip_frames: Process every nth frame
        use_ffmpeg: Use FFmpeg for encoding (better quality)
        crf: Quality (0-51, lower=better, 18-28 recommended)
        preset: Encoding speed (medium, fast, slow, etc.)
    """
    if use_ffmpeg and not check_ffmpeg_available():
        print("⚠ Warning: FFmpeg not found. Install with:")
        print("   - Windows: choco install ffmpeg  OR  download from https://ffmpeg.org")
        print("   - macOS:   brew install ffmpeg")
        print("Falling back to OpenCV VideoWriter (lower quality).\n")
        use_ffmpeg = False
    
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
    
    print(f"📹 Input video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    depth_cap = None
    if depth_video_path:
        depth_cap = cv2.VideoCapture(depth_video_path)
        if not depth_cap.isOpened():
            raise RuntimeError(f"Failed to open depth video: {depth_video_path}")
        depth_total = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if depth_total != total_frames:
            print(f"⚠ Warning: Depth video has {depth_total} frames, input has {total_frames}")
    
    out_width, out_height = IMAGE_SIZE, IMAGE_SIZE
    output_fps = fps / skip_frames
    
    if use_ffmpeg:
        writers = {
            'left': FFmpegVideoWriter(join(output_path, 'left_video.mp4'), out_width, out_height, output_fps, crf=crf, preset=preset),
            'warped': FFmpegVideoWriter(join(output_path, 'warped_video.mp4'), out_width, out_height, output_fps, crf=crf, preset=preset),
            'generated_right': FFmpegVideoWriter(join(output_path, 'generated_right_video.mp4'), out_width, out_height, output_fps, crf=crf, preset=preset),
            'disp': FFmpegVideoWriter(join(output_path, 'disp_video.mp4'), out_width, out_height, output_fps, crf=crf, preset=preset),
        }
        codec_info = f"H.264/libx264 (CRF={crf}, preset={preset})"
    else:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                print("⚠ Fallback to Motion JPEG (lower quality)")
        
        writers = {
            'left': cv2.VideoWriter(join(output_path, 'left_video.mp4'), fourcc, output_fps, (out_width, out_height)),
            'warped': cv2.VideoWriter(join(output_path, 'warped_video.mp4'), fourcc, output_fps, (out_width, out_height)),
            'generated_right': cv2.VideoWriter(join(output_path, 'generated_right_video.mp4'), fourcc, output_fps, (out_width, out_height)),
            'disp': cv2.VideoWriter(join(output_path, 'disp_video.mp4'), fourcc, output_fps, (out_width, out_height)),
        }
        codec_info = "OpenCV H.264 (mp4v)"
        
        for key, writer in writers.items():
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {key}")
    
    print(f"💾 Output codec: {codec_info}")
    print(f"Processing video (every {skip_frames} frame(s))...\n")
    
    frame_idx = 0
    processed_frames = 0
    
    try:
        pbar = tqdm(total=total_frames // skip_frames, unit='frame', desc='Processing')
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            if depth_cap:
                ret_depth, depth_frame_bgr = depth_cap.read()
                if not ret_depth:
                    print("⚠ Warning: Depth video ended prematurely")
                    break
                depth_frame_gray = cv2.cvtColor(depth_frame_bgr, cv2.COLOR_BGR2GRAY)
                depth_pil = Image.fromarray(depth_frame_gray)
                depth_pil = crop(depth_pil).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                depth = to_tensor(depth_pil).unsqueeze(0).float().to(DEVICE)
            else:
                depth = infer_depth_dam2(frame_bgr)
            
            if frame_idx % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = crop(frame_pil).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                
                outputs = process_frame(frame_pil, depth, scale_factor)
                
                for key, writer in writers.items():
                    writer.write(outputs[key])
                
                processed_frames += 1
                pbar.update(1)
            
            frame_idx += 1
        
        pbar.close()
    
    finally:
        cap.release()
        if depth_cap:
            depth_cap.release()
        for writer in writers.values():
            writer.release()
    
    print(f"\n✅ Processed {processed_frames} frames")
    print(f"📂 Output videos saved to: {output_path}/")
    print(f"   - left_video.mp4")
    print(f"   - warped_video.mp4")
    print(f"   - generated_right_video.mp4")
    print(f"   - disp_video.mp4")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate novel view from input video with depth estimation")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--depth_video_path", default=None, help="Path to depth video. If not provided, depth will be inferred per-frame.")
    parser.add_argument("--output", default="./vis", help="Output directory")
    parser.add_argument("--scale_factor", type=float, default=0.15, help="Disparity scaling factor")
    parser.add_argument("--skip_frames", type=int, default=1, help="Process every nth frame (default 1). Use >1 for faster processing.")
    parser.add_argument("--no_ffmpeg", action='store_true', help="Disable FFmpeg and use OpenCV for encoding instead.")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality (0-51, lower=better, 18-28 recommended)")
    parser.add_argument("--preset", default='medium', help="Encoding speed (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)")
    
    args = parser.parse_args()
    
    process_video(
        args.video_path,
        args.depth_video_path,
        args.output,
        scale_factor=args.scale_factor,
        skip_frames=args.skip_frames,
        use_ffmpeg=not args.no_ffmpeg,
        crf=args.crf,
        preset=args.preset
    )