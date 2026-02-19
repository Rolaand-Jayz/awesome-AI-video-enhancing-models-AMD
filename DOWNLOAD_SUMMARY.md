# Download Summary

## Successfully Downloaded Models (Local Only)

The following models have been downloaded to your local repository but are excluded from git due to their large file sizes:

### ✅ Available Locally (615 MB total)

1. **FBCNN** - `models/FBCNN/FBCNN_Color.pth` (275 MB)
   - JPEG artifact removal and image restoration
   
2. **SCUNet** - `models/SCUNet/scunet_color_real_psnr.pth` (69 MB)
   - Practical blind denoising
   
3. **DnCNN** - `models/DnCNN/DnCNN_color_blind.mat` (2.7 MB)
   - Classic denoising model
   
4. **Restormer** - `models/Restormer/Restormer_Motion_Deblur.pth` (100 MB)
   - Multi-task restoration (motion deblur)
   
5. **SwinIR (×2)** - 82 MB total
   - `models/SwinIR/SwinIR_real_sr_x4.pth` (65 MB)
   - `models/SwinIR/SwinIR_lightweight_sr_x4.pth` (17 MB)
   
6. **Real-ESRGAN (×3)** - 86.7 MB total
   - `models/Real-ESRGAN/RealESRGAN_x4plus_anime.pth` (18 MB)
   - `models/Real-ESRGAN/RealESRGAN_x4plus.pth` (64 MB)
   - `models/Real-ESRGAN/realesr-general-x4v3.pth` (4.7 MB)

## Models Requiring Manual Download

These models are hosted on platforms that require manual download (Google Drive, Baidu, HuggingFace with authentication):

### 📥 Manual Download Required

1. **SRVGGNet**
   - Download from: https://huggingface.co/Phips/2xHFA2kAVCCompact
   - Place in: `models/SRVGGNet/`

2. **RealDN** (REAL Video Enhancer Models)
   - Download from: https://github.com/TNTwise/real-video-enhancer-models/releases
   - Place in: `models/RealDN/`

3. **NAFNet (×3 variants)**
   - NAFNet-GoPro-width64: https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view
   - NAFNet-SIDD-width64: https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view
   - NAFNet-GoPro-width32: https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view
   - Place in: `models/NAFNet/`

4. **DAT** (Dual Aggregation Transformer)
   - Download folder: https://drive.google.com/drive/folders/1iBdf_-LVZuz_PAbFtuxSKd_11RL1YKxM
   - Recommended: DAT x2, DAT x4
   - Place in: `models/DAT/`

5. **RIFE (×3 versions)**
   - v4.6: https://drive.google.com/file/d/1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe/view
   - v4.14: https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view
   - v4.25 (Latest): https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view
   - Place in: `models/RIFE/`

## Special Cases

### CAS (Contrast Adaptive Sharpening)
- **Type**: Shader-based algorithm (not a neural network)
- **Source**: AMD FidelityFX - https://gpuopen.com/fidelityfx-cas/
- **Note**: Integrated into video players and renderers, not a downloadable model file

### FFmpeg
- **Type**: Software tool for video processing
- **Download**: https://ffmpeg.org/download.html
- **Note**: Not a neural network model, but essential for video frame extraction and encoding

### Real-ESRGAN NCNN
- **Type**: NCNN-optimized portable executable
- **Download**: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
- **Note**: Includes models and executables in one package

## Quick Start Guide

### For Models Already Downloaded

Models are stored locally in the `models/` directory structure. To use them:

1. Check the model files exist:
   ```bash
   ls -lh models/*/
   ```

2. Refer to the respective GitHub repositories for usage:
   - FBCNN: https://github.com/jiaxi-jiang/FBCNN
   - SCUNet: https://github.com/cszn/SCUNet
   - DnCNN: https://github.com/cszn/DnCNN
   - Restormer: https://github.com/swz30/Restormer
   - SwinIR: https://github.com/JingyunLiang/SwinIR
   - Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN

### To Download Remaining Models

1. **Automated download** (for models available on GitHub):
   ```bash
   chmod +x download_models.sh
   ./download_models.sh
   ```

2. **Manual download** (for Google Drive/HuggingFace models):
   - Follow the links in the "Manual Download Required" section above
   - Download the model files
   - Place them in the appropriate `models/` subdirectory

## Repository Structure

```
awesome-AI-video-enhancing-models-AMD/
├── README.md              # Main documentation
├── MODELS.md              # Detailed model information
├── DOWNLOAD_SUMMARY.md    # This file
├── download_models.sh     # Automated download script
├── .gitignore             # Excludes large model files
└── models/                # Model storage directory
    ├── FBCNN/             # ✅ Downloaded locally
    ├── SCUNet/            # ✅ Downloaded locally
    ├── SRVGGNet/          # ⚠️  Manual download required
    ├── RealDN/            # ⚠️  Manual download required
    ├── DnCNN/             # ✅ Downloaded locally
    ├── NAFNet/            # ⚠️  Manual download required (×3)
    ├── Restormer/         # ✅ Downloaded locally
    ├── SwinIR/            # ✅ Downloaded locally (×2)
    ├── DAT/               # ⚠️  Manual download required
    ├── CAS/               # ℹ️  Shader-based (N/A)
    ├── RIFE/              # ⚠️  Manual download required (×3)
    └── Real-ESRGAN/       # ✅ Downloaded locally (×3)
```

## Notes

- **Large files excluded from git**: Model files (.pth, .mat, .pkl) are excluded from version control due to their size
- **Local storage only**: Downloaded models are stored locally and not pushed to the repository
- **Total downloaded**: Approximately 615 MB of models are available locally
- **Platform limitations**: HuggingFace and Google Drive access may be restricted in some environments
- **Alternative sources**: For models requiring manual download, check if NCNN or ONNX versions are available

## Additional Resources

- **OpenModelDB**: https://openmodeldb.info/ - Community-trained models
- **Model Zoo Collections**: Many repositories include pre-trained models in their releases
- **NCNN Versions**: Check for NCNN ports for cross-platform compatibility

## Support

For issues with specific models, refer to their respective GitHub repositories linked in README.md and MODELS.md.
