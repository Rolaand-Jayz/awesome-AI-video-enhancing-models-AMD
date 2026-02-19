# Awesome AI Video Enhancing Models for AMD

A comprehensive collection of AI models for video enhancement optimized for AMD hardware. This repository includes various state-of-the-art models for video upscaling, denoising, deblurring, interpolation, and general video restoration.

## 📦 Included Models

### ✅ Downloaded Models (Ready to Use)

#### 1. **FBCNN** - Flexible Blind Convolutional Neural Network
- **Purpose**: JPEG artifact removal and image/video restoration
- **Location**: `models/FBCNN/FBCNN_Color.pth`
- **Size**: 275 MB
- **Source**: [GitHub - jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN)
- **Paper**: [ICCV 2021](https://arxiv.org/abs/2109.14348)

#### 2. **SCUNet** - Swin-Conv-UNet
- **Purpose**: Practical blind image/video denoising
- **Location**: `models/SCUNet/scunet_color_real_psnr.pth`
- **Size**: 69 MB
- **Source**: [GitHub - cszn/SCUNet](https://github.com/cszn/SCUNet)
- **Paper**: [CVPR 2022](https://arxiv.org/abs/2203.13278)

#### 3. **DnCNN** - Denoising CNN
- **Purpose**: Image/video denoising
- **Location**: `models/DnCNN/DnCNN_color_blind.mat`
- **Size**: 2.7 MB
- **Source**: [GitHub - cszn/DnCNN](https://github.com/cszn/DnCNN)
- **Paper**: [IEEE TIP 2017](https://arxiv.org/abs/1608.03981)

#### 4. **Restormer** - Efficient Transformer for High-Resolution Image Restoration
- **Purpose**: Motion deblurring, defocus deblurring, denoising, deraining
- **Location**: `models/Restormer/Restormer_Motion_Deblur.pth`
- **Size**: 100 MB
- **Source**: [GitHub - swz30/Restormer](https://github.com/swz30/Restormer)
- **Paper**: [CVPR 2022](https://arxiv.org/abs/2111.09881)

#### 5. **SwinIR** - Image Restoration Using Swin Transformer (×2 variants)
- **Purpose**: Super-resolution, denoising, JPEG artifact removal
- **Location**: 
  - `models/SwinIR/SwinIR_real_sr_x4.pth` (65 MB)
  - `models/SwinIR/SwinIR_lightweight_sr_x4.pth` (17 MB)
- **Source**: [GitHub - JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Paper**: [ICCV 2021 Workshop](https://arxiv.org/abs/2108.10257)

#### 6. **Real-ESRGAN** (×3 variants)
- **Purpose**: Real-world video/image super-resolution
- **Variants**:
  - **x4 Anime Video**: `models/Real-ESRGAN/RealESRGAN_x4plus_anime.pth` (18 MB)
  - **x4 General**: `models/Real-ESRGAN/RealESRGAN_x4plus.pth` (64 MB)
  - **x4 Artifact-Optimised**: `models/Real-ESRGAN/realesr-general-x4v3.pth` (4.7 MB)
- **Source**: [GitHub - xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Paper**: [ICCVW 2021](https://arxiv.org/abs/2107.10833)

### 📥 Models Requiring Manual Download

The following models require manual download due to hosting on platforms that may require authentication (Google Drive, Baidu, etc.):

#### 7. **SRVGGNet** - Compact Real-ESRGAN
- **Purpose**: Compact video upscaling
- **Download**: [HuggingFace - Phips/2xHFA2kAVCCompact](https://huggingface.co/Phips/2xHFA2kAVCCompact)
- **Installation**: Place `.pth` file in `models/SRVGGNet/`

#### 8. **RealDN** (REAL Video Enhancer)
- **Purpose**: Video interpolation, upscaling, denoising, decompression
- **Download**: 
  - Main repo: [GitHub - TNTwise/REAL-Video-Enhancer](https://github.com/TNTwise/REAL-Video-Enhancer)
  - Models: [GitHub - TNTwise/real-video-enhancer-models](https://github.com/TNTwise/real-video-enhancer-models/releases)
- **Installation**: Place models in `models/RealDN/`

#### 9. **NAFNet** - Nonlinear Activation Free Network (×3 variants recommended)
- **Purpose**: Image deblurring and denoising
- **Download**: 
  - NAFNet-GoPro-width64: [Google Drive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view)
  - NAFNet-SIDD-width64: [Google Drive](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view)
  - NAFNet-GoPro-width32: [Google Drive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view)
- **Source**: [GitHub - megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- **Paper**: [ECCV 2022](https://arxiv.org/abs/2204.04676)
- **Installation**: Place `.pth` files in `models/NAFNet/`

#### 10. **DAT** - Dual Aggregation Transformer
- **Purpose**: Image super-resolution
- **Download**: [Google Drive - Pretrained Models](https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM)
  - DAT x2
  - DAT x3
  - DAT x4
- **Source**: [GitHub - zhengchen1999/DAT](https://github.com/zhengchen1999/DAT)
- **Paper**: [ICCV 2023](https://arxiv.org/abs/2308.03364)
- **Installation**: Place `.pth` files in `models/DAT/`

#### 11. **CAS** (Content-Aware Scaling)
- **Purpose**: Video scaling
- **Note**: CAS is typically a real-time shader-based upscaling technique (AMD FidelityFX)
- **Source**: [AMD FidelityFX](https://gpuopen.com/fidelityfx-cas/)
- **Installation**: CAS is usually integrated into video players/renderers

#### 12. **RIFE** - Real-Time Intermediate Flow Estimation (×3 versions)
- **Purpose**: Video frame interpolation
- **Download**:
  - v4.6: [Google Drive](https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view)
  - v4.7: Contact maintainer or check releases
  - v4.14: [Google Drive](https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view)
  - v4.25 (Latest): [Google Drive](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view)
- **Source**: [GitHub - hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- **Paper**: [ECCV 2022](https://arxiv.org/abs/2011.06294)
- **Installation**: Extract and place `.pkl` files in `models/RIFE/`

#### 13. **FFmpeg**
- **Purpose**: Video processing tool (not a model)
- **Download**: [FFmpeg Official](https://ffmpeg.org/download.html)
- **Note**: FFmpeg is a command-line tool for video encoding/decoding, not a neural network model

#### 14. **Real-ESRGAN NCNN** (for AMD/Vulkan)
- **Purpose**: NCNN-optimized version for cross-platform deployment
- **Download**: [Real-ESRGAN-ncnn-vulkan Releases](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases)
- **Note**: This is a portable executable with models included

## 🚀 Quick Start

### Directory Structure
```
models/
├── FBCNN/
│   └── FBCNN_Color.pth
├── SCUNet/
│   └── scunet_color_real_psnr.pth
├── SRVGGNet/
│   └── (download required)
├── RealDN/
│   └── (download required)
├── DnCNN/
│   └── DnCNN_color_blind.mat
├── NAFNet/
│   └── (download required - 3 variants)
├── Restormer/
│   └── Restormer_Motion_Deblur.pth
├── SwinIR/
│   ├── SwinIR_real_sr_x4.pth
│   └── SwinIR_lightweight_sr_x4.pth
├── DAT/
│   └── (download required)
├── CAS/
│   └── (shader-based, see AMD FidelityFX)
├── RIFE/
│   └── (download required - 3 versions)
└── Real-ESRGAN/
    ├── RealESRGAN_x4plus.pth
    ├── RealESRGAN_x4plus_anime.pth
    └── realesr-general-x4v3.pth
```

### Usage Examples

Each model has different usage patterns. Refer to the respective GitHub repositories for detailed documentation.

## 📊 Model Comparison

| Model | Task | Speed | Quality | AMD Optimized |
|-------|------|-------|---------|---------------|
| FBCNN | Artifact Removal | Fast | High | ✓ |
| SCUNet | Denoising | Medium | Very High | ✓ |
| DnCNN | Denoising | Very Fast | Medium | ✓ |
| NAFNet | Deblurring/Denoising | Medium | Very High | ✓ |
| Restormer | Multi-task Restoration | Medium | Very High | ✓ |
| SwinIR | Super-Resolution | Medium | Very High | ✓ |
| DAT | Super-Resolution | Slow | Highest | ✓ |
| RIFE | Frame Interpolation | Fast | High | ✓ |
| Real-ESRGAN | Super-Resolution | Medium | Very High | ✓ |

## 🔧 AMD-Specific Optimizations

These models can be optimized for AMD GPUs using:
- **ROCm**: AMD's open-source platform for GPU computing
- **DirectML**: Microsoft's hardware-accelerated ML on Windows
- **ONNX Runtime**: Cross-platform inference with AMD GPU support
- **AMD FidelityFX**: Real-time upscaling shaders

## 📝 License

Each model has its own license. Please refer to the individual model repositories for licensing information.

## 🙏 Acknowledgments

This collection includes models from various research groups and open-source contributors. Please cite the original papers when using these models in research.

## 📚 Additional Resources

- [Video Enhancement Reading List](https://github.com/yulunzhang/video-enhancement)
- [Awesome Video Enhancement](https://github.com/topics/video-enhancement)
- [OpenModelDB](https://openmodeldb.info/) - Community-trained models
