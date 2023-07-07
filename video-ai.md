[🏠Home](README.md)

# Video
## Text to video generation
- [ModelScope Text to video synthesis](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis)
  -  [zeroscope v2 xl](https://huggingface.co/cerspense/zeroscope_v2_XL) Watermark free modelscope based video model generating high quality video at 1024x576 16:9, to be used with [text2video extension](https://github.com/kabachuha/sd-webui-text2video) for automatic1111
- [Nvidia VideoLDM: Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)
- [Potat1](https://huggingface.co/camenduru/potat1) , [colab](https://github.com/camenduru/text-to-video-synthesis-colab)
- [Phenaki](https://openreview.net/forum?id=vOEXS39nOF) multi minute text to video prompts with scene changes, [project page](https://phenaki.video/)

## Frame Interpolation (Temporal Interpolation)
- https://github.com/google-research/frame-interpolation
- https://github.com/ltkong218/ifrnet
- https://github.com/megvii-research/ECCV2022-RIFE

## Segmentation & Tracking
- [Segment and Track Anything](https://arxiv.org/abs/2305.06558v1), [code](https://github.com/z-x-yang/segment-and-track-anything). an innovative framework combining the Segment Anything Model (SAM) and DeAOT tracking model, enables precise, multimodal object tracking in video, demonstrating superior performance in benchmarks
- [Track Anything](https://arxiv.org/abs/2304.11968v2), [code](https://github.com/gaomingqi/track-anything). extends the Segment Anything Model (SAM) to achieve high-performance, interactive tracking and segmentation in videos with minimal human intervention, addressing SAM's limitations in consistent video segmentation
- [MAGVIT](https://magvit.cs.cmu.edu/) Single model for multiple video synthesis outperforming existing methods in quality and inference time, [code and models](https://github.com/MAGVIT/magvit), [paper](https://arxiv.org/pdf/2212.05199.pdf)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) Fast Segment Anything, a CNN trained achieving a comparable performance with the SAM method at 50× higher run-time speed.
- [SAM-PT](https://github.com/SysCV/sam-pt) Extending SAM to zero-shot video segmentation with point-based tracking, [paper](https://arxiv.org/abs/2307.01197)

## Super Resolution (Spacial Interpolation)
- https://github.com/researchmm/FTVSR
- https://github.com/picsart-ai-research/videoinr-continuous-space-time-super-resolution

## Spacio Temporal Interpolation
- https://github.com/llmpass/RSTT

## NeRF
- [Instant-ngp](https://github.com/NVlabs/instant-ngp) Train NeRFs in under 5 seconds on windows/linux with support for GPUs
- [NeRFstudio](https://github.com/nerfstudio-project/nerfstudio) A Collaboration Friendly Studio for NeRFs simplifying the process of creating, training, and testing NeRFs and supports web-based visualizer, benchmarks, and pipeline support.
- [Threestudio](https://github.com/threestudio-project/threestudio) A Framework for 3D Content Creation from Text Prompts, Single Images, and Few-Shot Images or text2image created single image to 3D
- [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) Zero-shot One Image to 3D Object for novel view synthesis and 3D reconstruction
- [localrf](https://github.com/facebookresearch/localrf) NeRFs for reconstructing large-scale stabilized scenes from shakey videos, [paper](https://localrf.github.io/localrf.pdf), [project page](https://localrf.github.io/)

## Deepfakes
- [roop](https://github.com/s0md3v/roop) one-click deepfake (face swap)

## Benchmarking
- [MSU Benchmarks](https://videoprocessing.ai/) collection of video processing benchmarks developed by the Video Processing Group at the Moscow State University
- [Video Super Resolution Benchmarks](https://paperswithcode.com/task/video-super-resolution)
- [Video Generation Benchmarks](https://paperswithcode.com/task/video-generation)
- [Video Frame Interpolation Benchmarks](https://paperswithcode.com/task/video-frame-interpolation)
