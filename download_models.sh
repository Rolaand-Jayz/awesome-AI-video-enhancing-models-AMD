#!/bin/bash
# Download script for AI video enhancement models
# Some models require manual download from Google Drive or Baidu due to authentication

set -e

echo "========================================="
echo "AI Video Enhancement Models Downloader"
echo "========================================="
echo ""

# Create model directories if they don't exist
mkdir -p models/{FBCNN,SCUNet,SRVGGNet,RealDN,DnCNN,NAFNet,Restormer,SwinIR,DAT,CAS,RIFE,Real-ESRGAN}

echo "Downloading models from GitHub..."
echo ""

# FBCNN
if [ ! -f "models/FBCNN/FBCNN_Color.pth" ]; then
    echo "Downloading FBCNN Color model..."
    wget -O models/FBCNN/FBCNN_Color.pth https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth
else
    echo "✓ FBCNN Color model already exists"
fi

# SCUNet
if [ ! -f "models/SCUNet/scunet_color_real_psnr.pth" ]; then
    echo "Downloading SCUNet model..."
    wget -O models/SCUNet/scunet_color_real_psnr.pth https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth
else
    echo "✓ SCUNet model already exists"
fi

# DnCNN
if [ ! -f "models/DnCNN/DnCNN_color_blind.mat" ]; then
    echo "Downloading DnCNN model..."
    wget -O models/DnCNN/DnCNN_color_blind.mat https://github.com/cszn/DnCNN/raw/master/model/GD_Color_Blind.mat
else
    echo "✓ DnCNN model already exists"
fi

# Restormer
if [ ! -f "models/Restormer/Restormer_Motion_Deblur.pth" ]; then
    echo "Downloading Restormer Motion Deblur model..."
    wget -O models/Restormer/Restormer_Motion_Deblur.pth https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth
else
    echo "✓ Restormer model already exists"
fi

# SwinIR
if [ ! -f "models/SwinIR/SwinIR_real_sr_x4.pth" ]; then
    echo "Downloading SwinIR Real SR x4 model..."
    wget -O models/SwinIR/SwinIR_real_sr_x4.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
else
    echo "✓ SwinIR Real SR x4 model already exists"
fi

if [ ! -f "models/SwinIR/SwinIR_lightweight_sr_x4.pth" ]; then
    echo "Downloading SwinIR Lightweight SR x4 model..."
    wget -O models/SwinIR/SwinIR_lightweight_sr_x4.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth
else
    echo "✓ SwinIR Lightweight SR x4 model already exists"
fi

# Real-ESRGAN
if [ ! -f "models/Real-ESRGAN/RealESRGAN_x4plus_anime.pth" ]; then
    echo "Downloading Real-ESRGAN x4 Anime model..."
    wget -O models/Real-ESRGAN/RealESRGAN_x4plus_anime.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
else
    echo "✓ Real-ESRGAN x4 Anime model already exists"
fi

if [ ! -f "models/Real-ESRGAN/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading Real-ESRGAN x4 General model..."
    wget -O models/Real-ESRGAN/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
else
    echo "✓ Real-ESRGAN x4 General model already exists"
fi

if [ ! -f "models/Real-ESRGAN/realesr-general-x4v3.pth" ]; then
    echo "Downloading Real-ESRGAN Artifact-Optimized model..."
    wget -O models/Real-ESRGAN/realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
else
    echo "✓ Real-ESRGAN Artifact-Optimized model already exists"
fi

echo ""
echo "========================================="
echo "GitHub models downloaded successfully!"
echo "========================================="
echo ""
echo "⚠️  Manual Download Required:"
echo ""
echo "The following models require manual download from Google Drive/Baidu:"
echo ""
echo "1. NAFNet (×3 variants):"
echo "   - NAFNet-GoPro-width64: https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view"
echo "   - NAFNet-SIDD-width64: https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view"
echo "   - NAFNet-GoPro-width32: https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view"
echo "   Place in: models/NAFNet/"
echo ""
echo "2. DAT (Dual Aggregation Transformer):"
echo "   - Download from: https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM"
echo "   Place in: models/DAT/"
echo ""
echo "3. RIFE (×3 versions):"
echo "   - v4.6: https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view"
echo "   - v4.14: https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view"
echo "   - v4.25 (Latest): https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view"
echo "   Place in: models/RIFE/"
echo ""
echo "4. SRVGGNet:"
echo "   - Download from: https://huggingface.co/Phips/2xHFA2kAVCCompact"
echo "   Place in: models/SRVGGNet/"
echo ""
echo "5. RealDN (REAL Video Enhancer):"
echo "   - Download models from: https://github.com/TNTwise/real-video-enhancer-models/releases"
echo "   Place in: models/RealDN/"
echo ""
echo "See README.md for more details and usage instructions."
echo "========================================="
