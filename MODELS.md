# AI Video Enhancement Models - Detailed Information

## Table of Contents
- [FBCNN](#fbcnn)
- [SCUNet](#scunet)
- [SRVGGNet](#srvggnet)
- [RealDN](#realdn)
- [DnCNN](#dncnn)
- [NAFNet](#nafnet)
- [Restormer](#restormer)
- [SwinIR](#swinir)
- [DAT](#dat)
- [CAS](#cas)
- [RIFE](#rife)
- [Real-ESRGAN](#real-esrgan)
- [FFmpeg](#ffmpeg)

---

## FBCNN

**Flexible Blind Convolutional Neural Network**

### Overview
FBCNN is designed for blind JPEG artifact removal and general image restoration. It can handle various compression quality factors without requiring prior knowledge.

### Key Features
- Flexible handling of different JPEG quality factors
- Single model for multiple compression levels
- High-quality artifact removal
- Can be applied frame-by-frame for video

### Technical Specs
- **Architecture**: CNN-based with flexible blind estimation
- **Input**: RGB images (can process video frames)
- **Output**: Artifact-free images
- **Model Size**: 275 MB

### Download Status
✅ **Downloaded** - `models/FBCNN/FBCNN_Color.pth`

### References
- **Paper**: [Towards Flexible Blind JPEG Artifacts Removal (ICCV 2021)](https://arxiv.org/abs/2109.14348)
- **Repository**: https://github.com/jiaxi-jiang/FBCNN

---

## SCUNet

**Swin-Conv-UNet Network**

### Overview
SCUNet combines Swin Transformer blocks with convolutional layers for practical blind image denoising, achieving state-of-the-art results.

### Key Features
- Blind denoising (no need to know noise level)
- Real-world noise handling
- Excellent PSNR/SSIM metrics
- Efficient architecture

### Technical Specs
- **Architecture**: Swin Transformer + U-Net
- **Input**: RGB images
- **Output**: Denoised images
- **Model Size**: 69 MB

### Download Status
✅ **Downloaded** - `models/SCUNet/scunet_color_real_psnr.pth`

### References
- **Paper**: [Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis (CVPR 2022)](https://arxiv.org/abs/2203.13278)
- **Repository**: https://github.com/cszn/SCUNet

---

## SRVGGNet

**Super-Resolution VGG Network (Compact)**

### Overview
A compact version of Real-ESRGAN designed for efficient video upscaling with lower computational requirements.

### Key Features
- Lightweight architecture
- 2x upscaling optimized for anime/video
- Handles AVC (h264) compression artifacts
- Fast inference

### Technical Specs
- **Architecture**: Compact VGG-style network
- **Input**: Low-resolution video frames
- **Output**: 2x upscaled frames
- **Model Size**: ~10-15 MB

### Download Status
⚠️ **Manual Download Required**
- HuggingFace: https://huggingface.co/Phips/2xHFA2kAVCCompact
- Place in: `models/SRVGGNet/`

### References
- **Repository**: https://github.com/xinntao/Real-ESRGAN

---

## RealDN

**REAL Video Enhancer (Denoising)**

### Overview
Part of the REAL Video Enhancer suite, providing comprehensive video enhancement including denoising, upscaling, and interpolation.

### Key Features
- Multi-task video enhancement
- Denoising, upscaling, decompression
- GPU-accelerated
- User-friendly interface available

### Technical Specs
- **Architecture**: Multiple models for different tasks
- **Input**: Video files or frame sequences
- **Output**: Enhanced video
- **Model Size**: Varies by model

### Download Status
⚠️ **Manual Download Required**
- Main Repo: https://github.com/TNTwise/REAL-Video-Enhancer
- Models: https://github.com/TNTwise/real-video-enhancer-models/releases
- Place in: `models/RealDN/`

### References
- **Repository**: https://github.com/TNTwise/REAL-Video-Enhancer

---

## DnCNN

**Denoising Convolutional Neural Network**

### Overview
A classic and efficient CNN-based denoising model that remains popular due to its simplicity and effectiveness.

### Key Features
- Fast inference
- Effective for Gaussian noise
- Lightweight model
- Both blind and non-blind variants

### Technical Specs
- **Architecture**: Deep CNN (17 layers)
- **Input**: Grayscale or RGB images
- **Output**: Denoised images
- **Model Size**: 2.7 MB (MATLAB format)

### Download Status
✅ **Downloaded** - `models/DnCNN/DnCNN_color_blind.mat`

### References
- **Paper**: [Beyond a Gaussian Denoiser (IEEE TIP 2017)](https://arxiv.org/abs/1608.03981)
- **Repository**: https://github.com/cszn/DnCNN

---

## NAFNet

**Nonlinear Activation Free Network**

### Overview
NAFNet achieves state-of-the-art results without using traditional activation functions (ReLU, Sigmoid, etc.), proving simpler can be better.

### Key Features
- No nonlinear activation functions
- State-of-the-art deblurring performance
- Excellent denoising results
- Computationally efficient
- Multiple variants available

### Technical Specs
- **Architecture**: Activation-free baseline
- **Input**: Blurred/noisy images
- **Output**: Restored images
- **Model Variants**: 
  - NAFNet-GoPro-width64 (deblurring)
  - NAFNet-SIDD-width64 (denoising)
  - NAFNet-GoPro-width32 (lightweight)

### Download Status
⚠️ **Manual Download Required** (×3 variants)
- NAFNet-GoPro-width64: https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view
- NAFNet-SIDD-width64: https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view
- NAFNet-GoPro-width32: https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view
- Place in: `models/NAFNet/`

### References
- **Paper**: [Simple Baselines for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)
- **Repository**: https://github.com/megvii-research/NAFNet

---

## Restormer

**Efficient Transformer for High-Resolution Image Restoration**

### Overview
Restormer is a multi-task restoration transformer that excels at motion deblurring, defocus deblurring, denoising, and deraining.

### Key Features
- Multi-task restoration
- Transformer-based architecture
- High-resolution support
- Excellent quality metrics

### Technical Specs
- **Architecture**: Transformer with multi-Dconv head
- **Tasks**: Deblurring, Denoising, Deraining
- **Input**: Degraded images
- **Output**: Restored images
- **Model Size**: 100 MB

### Download Status
✅ **Downloaded** - `models/Restormer/Restormer_Motion_Deblur.pth`

### References
- **Paper**: [Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)](https://arxiv.org/abs/2111.09881)
- **Repository**: https://github.com/swz30/Restormer

---

## SwinIR

**Image Restoration Using Swin Transformer**

### Overview
SwinIR uses Swin Transformer for various image restoration tasks including super-resolution, denoising, and JPEG artifact removal.

### Key Features
- Swin Transformer architecture
- Multiple restoration tasks
- Excellent super-resolution quality
- Available in multiple scales (x2, x3, x4)

### Technical Specs
- **Architecture**: Swin Transformer
- **Tasks**: Super-resolution, Denoising, JPEG compression
- **Input**: Low-quality images
- **Output**: High-quality restored images
- **Model Variants**:
  - Real SR x4 (65 MB) - For real-world super-resolution
  - Lightweight SR x4 (17 MB) - Faster, smaller model

### Download Status
✅ **Downloaded** (×2 variants)
- `models/SwinIR/SwinIR_real_sr_x4.pth`
- `models/SwinIR/SwinIR_lightweight_sr_x4.pth`

### References
- **Paper**: [SwinIR: Image Restoration Using Swin Transformer (ICCV 2021)](https://arxiv.org/abs/2108.10257)
- **Repository**: https://github.com/JingyunLiang/SwinIR

---

## DAT

**Dual Aggregation Transformer**

### Overview
DAT aggregates features across both spatial and channel dimensions for superior image super-resolution performance.

### Key Features
- Dual aggregation mechanism
- State-of-the-art super-resolution
- Efficient transformer design
- Multiple scaling factors (x2, x3, x4)

### Technical Specs
- **Architecture**: Transformer with dual aggregation
- **Task**: Super-resolution
- **Input**: Low-resolution images
- **Output**: High-resolution images
- **Available Models**: DAT, DAT-S, DAT-2, DAT-light

### Download Status
⚠️ **Manual Download Required**
- Download: https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM
- Recommended: DAT x2, DAT x4
- Place in: `models/DAT/`

### References
- **Paper**: [Dual Aggregation Transformer for Image Super-Resolution (ICCV 2023)](https://arxiv.org/abs/2308.03364)
- **Repository**: https://github.com/zhengchen1999/DAT

---

## CAS

**Contrast Adaptive Sharpening (AMD FidelityFX)**

### Overview
CAS is a real-time, shader-based upscaling and sharpening technique from AMD's FidelityFX suite, designed for gaming and real-time applications.

### Key Features
- Real-time performance
- Shader-based (runs on GPU)
- Adaptive contrast enhancement
- Low overhead
- AMD GPU optimized

### Technical Specs
- **Type**: Shader-based algorithm (not a neural network)
- **Platform**: AMD GPUs, DirectX, Vulkan
- **Usage**: Integrated into video players, games, renderers

### Download Status
ℹ️ **Not Applicable** (Shader-based, not a downloadable model)
- Implementation available at: https://gpuopen.com/fidelityfx-cas/
- Often integrated into video players like mpv, VLC (with plugins)

### References
- **Source**: https://gpuopen.com/fidelityfx-cas/
- **Documentation**: AMD FidelityFX SDK

---

## RIFE

**Real-Time Intermediate Flow Estimation**

### Overview
RIFE is a practical video frame interpolation model that can generate smooth slow-motion or increase frame rates in real-time.

### Key Features
- Real-time frame interpolation
- Multiple versions optimized for different scenarios
- Excellent motion handling
- Wide adoption in video software

### Technical Specs
- **Architecture**: Flow-based interpolation network
- **Task**: Frame interpolation (2x, 4x, 8x, etc.)
- **Input**: Video frames
- **Output**: Interpolated frames
- **Model Versions**:
  - v4.6: Stable, reliable
  - v4.14: Improved quality
  - v4.25: Latest, best quality

### Download Status
⚠️ **Manual Download Required** (×3 versions)
- v4.6: https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view
- v4.14: https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view
- v4.25 (Latest): https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view
- Place in: `models/RIFE/`

### References
- **Paper**: [Real-Time Intermediate Flow Estimation for Video Frame Interpolation (ECCV 2022)](https://arxiv.org/abs/2011.06294)
- **Repository**: https://github.com/hzwer/Practical-RIFE

---

## Real-ESRGAN

**Real-World Enhanced Super-Resolution GAN**

### Overview
Real-ESRGAN is designed for practical real-world super-resolution, handling various degradations like compression artifacts, blur, and noise.

### Key Features
- Real-world degradation handling
- Multiple specialized variants
- High-quality upscaling (2x, 4x)
- Anime and general image support

### Technical Specs
- **Architecture**: ESRGAN with improved training strategy
- **Task**: Super-resolution
- **Scaling**: 2x, 4x
- **Model Variants**:
  - x4 Anime (18 MB): Optimized for anime/cartoon content
  - x4 General (64 MB): General-purpose upscaling
  - x4 Artifact-Optimised (4.7 MB): Compact, artifact suppression

### Download Status
✅ **Downloaded** (×3 variants)
- `models/Real-ESRGAN/RealESRGAN_x4plus_anime.pth`
- `models/Real-ESRGAN/RealESRGAN_x4plus.pth`
- `models/Real-ESRGAN/realesr-general-x4v3.pth`

### Additional Format
📦 **NCNN Version Available**
- Real-ESRGAN-ncnn-vulkan: Cross-platform NCNN implementation
- Download: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
- Includes portable executables for Windows/Linux/macOS

### References
- **Paper**: [Real-ESRGAN: Training Real-World Blind Super-Resolution (ICCVW 2021)](https://arxiv.org/abs/2107.10833)
- **Repository**: https://github.com/xinntao/Real-ESRGAN

---

## FFmpeg

**Fast Forward Moving Picture Experts Group**

### Overview
FFmpeg is not a model but a comprehensive multimedia framework for video/audio processing, encoding, decoding, and streaming.

### Key Features
- Universal video/audio codec support
- Video encoding/decoding
- Format conversion
- Filtering and effects
- Hardware acceleration support

### Technical Specs
- **Type**: Command-line tool / Library
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **GPU Support**: NVENC, VAAPI, AMD VCE/AMF

### Download Status
ℹ️ **Not a Model** - Standalone software
- Download: https://ffmpeg.org/download.html
- AMD GPU encoding: Use `-c:v h264_amf` or `-c:v hevc_amf`

### Usage with AI Models
FFmpeg is typically used alongside AI models for:
- Extracting frames from video
- Encoding processed frames back to video
- Format conversion
- Hardware-accelerated encoding/decoding

### References
- **Official Site**: https://ffmpeg.org/
- **Documentation**: https://ffmpeg.org/documentation.html

---

## Summary

### Downloaded Models (Ready to Use)
1. ✅ FBCNN - 275 MB
2. ✅ SCUNet - 69 MB
3. ✅ DnCNN - 2.7 MB
4. ✅ Restormer - 100 MB
5. ✅ SwinIR (×2) - 82 MB total
6. ✅ Real-ESRGAN (×3) - 86.7 MB total

**Total Downloaded: ~615 MB**

### Requires Manual Download
1. ⚠️ SRVGGNet
2. ⚠️ RealDN
3. ⚠️ NAFNet (×3)
4. ⚠️ DAT
5. ⚠️ RIFE (×3)

### Not Applicable
1. ℹ️ CAS (Shader-based)
2. ℹ️ FFmpeg (Software tool)

---

## License Information

Each model has its own license. Please check the individual repositories for specific licensing terms. Most models are released under:
- MIT License
- Apache 2.0
- Creative Commons licenses

Always cite the original papers when using these models in academic work.
