import os
import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def enhance_image(input_image_path, output_dir="static/outputs"):
    model_path = os.path.join("models", "RealESRGAN_x4plus.pth")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    # ✅ Use tiling to reduce memory usage
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=128,              # 🧠 Enables tiling mode (safe for RAM)
        tile_pad=10,
        pre_pad=0,
        half=False             # Don't use FP16
    )

    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    output, _ = upsampler.enhance(img, outscale=4)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"enhanced_{os.path.basename(input_image_path)}"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, output)

    # ✅ Ensure file was saved
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Failed to save enhanced image at {save_path}")

    return save_path.replace("\\", "/")
