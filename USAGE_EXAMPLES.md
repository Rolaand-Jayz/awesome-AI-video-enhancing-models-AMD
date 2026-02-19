# Usage Examples

This document provides practical examples for using the AI video enhancement models in this repository.

## Prerequisites

- Python 3.8+
- PyTorch 1.8+ with CUDA/ROCm support (for AMD GPUs)
- FFmpeg (for video processing)
- Git LFS (optional, for large file handling)

## General Workflow

Most video enhancement tasks follow this pattern:

1. **Extract frames** from video using FFmpeg
2. **Process frames** with AI model
3. **Reassemble video** from processed frames

## AMD GPU Setup

### ROCm Installation (Linux)

```bash
# Install ROCm for AMD GPUs
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_5.4.50400-1_all.deb
sudo apt install ./amdgpu-install_5.4.50400-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify ROCm installation
rocm-smi
```

### PyTorch with ROCm

```bash
# Install PyTorch with ROCm support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Model-Specific Examples

### 1. Real-ESRGAN - Video Upscaling

```bash
# Clone Real-ESRGAN repository
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop

# Copy model to expected location
mkdir -p experiments/pretrained_models
cp ../models/Real-ESRGAN/RealESRGAN_x4plus.pth experiments/pretrained_models/

# Upscale video (extracts frames automatically)
python inference_realesrgan_video.py -i input_video.mp4 -o output_video.mp4 \
    -n RealESRGAN_x4plus -s 4 --suffix out

# For anime content, use:
python inference_realesrgan_video.py -i anime.mp4 -o anime_upscaled.mp4 \
    -n RealESRGAN_x4plus_anime_6B -s 4
```

### 2. RIFE - Frame Interpolation

```bash
# Clone RIFE repository
git clone https://github.com/hzwer/Practical-RIFE.git
cd Practical-RIFE
pip install -r requirements.txt

# Copy model (if manually downloaded)
mkdir -p train_log
cp ../models/RIFE/flownet-v4.25.pkl train_log/

# Interpolate video (2x frame rate)
python inference_video.py --multi=2 --video=input.mp4

# 4x interpolation
python inference_video.py --multi=4 --video=input.mp4

# For 4K videos, use scaling
python inference_video.py --multi=2 --video=4k_input.mp4 --scale=0.5
```

### 3. SwinIR - Super Resolution

```bash
# Clone SwinIR repository
git clone https://github.com/JingyunLiang/SwinIR.git
cd SwinIR

# Copy models
mkdir -p model_zoo/swinir
cp ../models/SwinIR/*.pth model_zoo/swinir/

# Process images
python main_test_swinir.py --task real_sr --scale 4 \
    --model_path model_zoo/swinir/SwinIR_real_sr_x4.pth \
    --folder_lq testsets/RealSRSet+5images
```

### 4. SCUNet - Denoising

```bash
# Clone KAIR repository (contains SCUNet)
git clone https://github.com/cszn/KAIR.git
cd KAIR

# Copy model
mkdir -p model_zoo
cp ../models/SCUNet/scunet_color_real_psnr.pth model_zoo/

# Denoise images
python main_test_scunet.py --model_name scunet_color_real_psnr \
    --testset_name your_noisy_images \
    --noise_level_img 15
```

### 5. NAFNet - Deblurring/Denoising

```bash
# Clone NAFNet repository
git clone https://github.com/megvii-research/NAFNet.git
cd NAFNet
pip install -r requirements.txt
python setup.py develop

# Copy models (after manual download)
mkdir -p experiments/pretrained_models
cp ../models/NAFNet/*.pth experiments/pretrained_models/

# For deblurring
python basicsr/test.py -opt options/test/NAFNet-width64.yml

# For denoising  
python basicsr/test.py -opt options/test/SIDD/NAFNet-width64.yml
```

### 6. Restormer - Motion Deblurring

```bash
# Clone Restormer repository
git clone https://github.com/swz30/Restormer.git
cd Restormer

# Copy model
mkdir -p Motion_Deblurring/pretrained_models
cp ../models/Restormer/Restormer_Motion_Deblur.pth Motion_Deblurring/pretrained_models/

# Process images
python demo.py --task Motion_Deblurring \
    --input_dir ./test_images/ \
    --result_dir ./results/
```

## Video Processing Pipeline Examples

### Complete Video Enhancement Pipeline

```bash
#!/bin/bash
# complete_enhancement.sh - Full video enhancement pipeline

INPUT_VIDEO="input.mp4"
OUTPUT_VIDEO="enhanced_output.mp4"
TEMP_DIR="temp_frames"

# Step 1: Extract frames
mkdir -p $TEMP_DIR/input
ffmpeg -i $INPUT_VIDEO $TEMP_DIR/input/%06d.png

# Step 2: Upscale with Real-ESRGAN
cd Real-ESRGAN
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../$TEMP_DIR/input \
    -o ../$TEMP_DIR/upscaled -s 4
cd ..

# Step 3: Denoise with SCUNet
cd KAIR
python main_test_scunet.py --model_name scunet_color_real_psnr \
    --testset_name ../$TEMP_DIR/upscaled --noise_level_img 10
cd ..

# Step 4: Reassemble video
ffmpeg -framerate 30 -i $TEMP_DIR/upscaled/%06d.png -c:v libx264 -pix_fmt yuv420p $OUTPUT_VIDEO

# Cleanup
rm -rf $TEMP_DIR
```

### Frame Interpolation Pipeline

```bash
#!/bin/bash
# interpolate_and_upscale.sh

INPUT="30fps_video.mp4"
INTERPOLATED="60fps_temp.mp4"
OUTPUT="60fps_upscaled.mp4"

# Step 1: Interpolate to 60fps
cd Practical-RIFE
python inference_video.py --multi=2 --video=../$INPUT --output=../$INTERPOLATED
cd ..

# Step 2: Upscale interpolated video
cd Real-ESRGAN  
python inference_realesrgan_video.py -i ../$INTERPOLATED -o ../$OUTPUT \
    -n RealESRGAN_x4plus -s 4
cd ..
```

## AMD-Specific Optimizations

### Using ROCm with PyTorch

```python
import torch

# Check ROCm availability
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Use AMD GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### DirectML (Windows)

```bash
# Install DirectML backend for Windows AMD GPUs
pip install torch-directml

# Use in Python
import torch_directml
device = torch_directml.device()
model = model.to(device)
```

## Batch Processing

### Process Multiple Videos

```bash
#!/bin/bash
# batch_upscale.sh

for video in *.mp4; do
    echo "Processing $video..."
    python inference_realesrgan_video.py \
        -i "$video" \
        -o "upscaled_${video}" \
        -n RealESRGAN_x4plus \
        -s 4
done
```

### Parallel Frame Processing

```python
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def process_frame(frame_path):
    # Your frame processing logic here
    pass

def batch_process_frames(input_dir, num_workers=4):
    frame_paths = list(Path(input_dir).glob('*.png'))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_frame, frame_paths)
```

## Performance Tips

1. **Use half-precision (FP16)** for faster processing:
   ```python
   model = model.half()
   input_tensor = input_tensor.half()
   ```

2. **Batch processing** for efficiency:
   ```python
   # Process multiple frames at once
   batch_size = 4
   for i in range(0, len(frames), batch_size):
       batch = frames[i:i+batch_size]
       outputs = model(batch)
   ```

3. **Tile processing** for large images:
   ```bash
   # Use tile mode to handle large images with limited VRAM
   python inference_realesrgan.py --tile=400 --tile_pad=10
   ```

4. **GPU memory management**:
   ```python
   torch.cuda.empty_cache()  # Clear cache between batches
   ```

## Common Issues & Solutions

### Out of Memory (OOM)

```bash
# Solution 1: Use smaller tile size
--tile=256 --tile_pad=10

# Solution 2: Process at lower resolution first
--scale=0.5

# Solution 3: Use FP16
model = model.half()
```

### Slow Processing on AMD GPU

```bash
# Enable ROCm optimizations
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
export PYTORCH_ROCM_ARCH="gfx1030"      # Adjust for your architecture
```

### Color Space Issues

```bash
# Ensure correct color space conversion
ffmpeg -i input.mp4 -pix_fmt rgb24 frames/%06d.png
```

## Advanced Usage

### Custom Model Fine-tuning

```python
# Example: Fine-tune Real-ESRGAN for specific content
from basicsr.train import train_pipeline

# Modify config file
opt = parse_options('options/train/train_realesrgan_x4plus.yml')
opt['datasets']['train']['dataroot_gt'] = 'your_hr_images'
opt['datasets']['train']['dataroot_lq'] = 'your_lr_images'

# Train
train_pipeline(opt)
```

### Ensemble Methods

```python
# Combine multiple models for better results
outputs = []
outputs.append(model1(input))
outputs.append(model2(input))
final_output = torch.mean(torch.stack(outputs), dim=0)
```

## References

- [Real-ESRGAN Documentation](https://github.com/xinntao/Real-ESRGAN)
- [RIFE Documentation](https://github.com/hzwer/Practical-RIFE)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

For more examples and detailed usage instructions, refer to the individual model repositories linked in README.md.
