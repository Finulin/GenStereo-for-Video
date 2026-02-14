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

def get_upsampler(scale=4):
    """Initialize and return Real-ESRGAN upsampler."""
    global _upsampler
    
    if _upsampler is not None:
        return _upsampler
    
    if not REALESRGAN_AVAILABLE:
        raise ImportError(
            "Real-ESRGAN not installed. Install with:\n"
            "  pip install basicsr realesrgan\n"
            "Or for faster installation:\n"
            "  pip install basicsr\n"
            "  pip install git+https://github.com/xinntao/Real-ESRGAN.git"
        )
    
    # Model paths
    esrgan_dir = 'checkpoints/esrgan'
    os.makedirs(esrgan_dir, exist_ok=True)
    
    if scale == 2:
        model_name = 'RealESRGAN_x2plus'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:  # scale 4
        model_name = 'RealESRGAN_x4plus'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model_path = join(esrgan_dir, f'{model_name}.pth')
    
    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"Downloading Real-ESRGAN model: {model_name}...")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded to: {model_path}")
    
    # Determine device for upscaler
    upscaler_device = DEVICE
    if DEVICE == 'mps':
        # Real-ESRGAN may have issues with MPS, fall back to CPU on Mac if needed
        upscaler_device = 'cpu'
    
    _upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,  # 0 = no tiling, use full image (may need more VRAM)
        tile_pad=10,
        pre_pad=0,
        half=False,  # Use float32 for better compatibility
        device=upscaler_device
    )
    
    return _upsampler

def upscale_image(img, scale=4):
    """Upscale image using Real-ESRGAN.
    
    Args:
        img: PIL Image or numpy array (BGR)
        scale: Upscaling factor (2 or 4)
    
    Returns:
        PIL Image (upscaled)
    """
    upsampler = get_upsampler(scale)
    
    # Convert PIL to numpy BGR if needed
    if isinstance(img, Image.Image):
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img_np = img
    
    # Upscale
    output, _ = upsampler.enhance(img_np, outscale=scale)
    
    # Convert back to PIL
    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    return output_pil

def calculate_output_size(original_size: tuple, max_size: int) -> tuple:
    """Calculate output size keeping aspect ratio."""
    W, H = original_size
    
    if W >= H:
        new_width = max_size
        new_height = int(H * (max_size / W))
    else:
        new_height = max_size
        new_width = int(W * (max_size / H))
    
    # Ensure dimensions are divisible by 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # Ensure minimum size
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)
    
    return (new_width, new_height)

def load_image_and_depth(image_path: str, depth_path: str):
    """Load image and depth map. Returns (processed_image, depth_tensor, original_size, output_size)."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    output_size = calculate_output_size(original_size, IMAGE_SIZE)
    
    # Resize to square for model processing
    square_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Load and resize depth map
    depth_map = Image.open(depth_path).convert('L')
    depth_map = depth_map.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    depth_tensor = to_tensor(depth_map).unsqueeze(0).float().to(DEVICE)
    
    return square_image, depth_tensor, original_size, output_size

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def generate_novel_view(image, depth, output_dir, basename, original_size, output_size, scale_factor=0.05, save_all=False, upscale=0):
    """Generate novel view and save outputs."""
    output_path = join(output_dir, basename)
    os.makedirs(output_path, exist_ok=True)

    disparity = normalize_disp(depth) * scale_factor * IMAGE_SIZE

    renders = genstereo_nvs(src_image=image, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    
    with torch.no_grad():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
    
    # Resize outputs to target output size (restoring aspect ratio)
    out_w, out_h = output_size
    fusion_image_pil = to_pil_image(fusion_image[0])
    fusion_image_pil = fusion_image_pil.resize((out_w, out_h), Image.BILINEAR)
    
    # Load original image and resize to output size
    original_image = image.resize((out_w, out_h), Image.BILINEAR)
    
    # Apply upscaling if requested
    if upscale > 0:
        print(f"Upscaling images {upscale}x...")
        original_image = upscale_image(original_image, upscale)
        fusion_image_pil = upscale_image(fusion_image_pil, upscale)
        out_w, out_h = original_image.size
    
    # Save standard outputs (left.png and generated_right.png)
    original_image.save(join(output_path, 'left.png'))
    fusion_image_pil.save(join(output_path, 'generated_right.png'))
    
    # Save additional outputs if requested
    if save_all:
        warped_pil = to_pil_image(warped[0])
        warped_pil = warped_pil.resize((out_w // upscale if upscale > 0 else out_w, out_h // upscale if upscale > 0 else out_h), Image.BILINEAR)
        
        depth_np = depth.squeeze().cpu().numpy()
        disp_vis = cv2.applyColorMap((normalize_disp(depth_np) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        disp_vis = cv2.resize(disp_vis, (out_w // upscale if upscale > 0 else out_w, out_h // upscale if upscale > 0 else out_h))
        
        if upscale > 0:
            warped_pil = upscale_image(warped_pil, upscale)
            disp_vis, _ = get_upsampler(upscale).enhance(disp_vis, outscale=upscale)
        
        cv2.imwrite(join(output_path, 'disp.png'), disp_vis)
        warped_pil.save(join(output_path, 'warped.png'))
    
    return (out_w, out_h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate novel view from input image with depth map")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("depth_path", help="Path to depth map (grayscale image)")
    parser.add_argument("--output", default="./vis", help="Output directory")
    parser.add_argument("--scale_factor", type=float, default=0.05, help="Disparity scaling factor")
    parser.add_argument("--save_all", action='store_true', help="Save all outputs (disp.png, warped.png) in addition to left.png and generated_right.png")
    parser.add_argument("--upscale", type=int, choices=[0, 2, 4], default=0, help="Upscale output images by factor (0=no upscaling, 2=2x, 4=4x). Requires: pip install basicsr realesrgan")
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
    
    base_name = splitext(basename(args.image_path))[0]
    
    print(f"Loading image from {args.image_path}")
    print(f"Loading depth map from {args.depth_path}")
    img, depth, orig_size, out_size = load_image_and_depth(args.image_path, args.depth_path)
    
    print(f"Original size: {orig_size[0]}x{orig_size[1]} px")
    print(f"Output width:  {out_size[0]} px")
    print(f"Output height: {out_size[1]} px")
    print(f"Scale factor:  {args.scale_factor}")
    print("")
    
    final_size = generate_novel_view(img, depth, args.output, base_name, orig_size, out_size, args.scale_factor, args.save_all, args.upscale)
    
    saved_files = "left.png, generated_right.png"
    if args.save_all:
        saved_files += ", disp.png, warped.png"
    print(f"Done! Saved: {saved_files}")
    if args.upscale > 0:
        print(f"Final size:   {final_size[0]}x{final_size[1]} px (upscaled {args.upscale}x)")
    print(f"Output directory: {join(args.output, base_name)}/")