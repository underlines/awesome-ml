[🏠Home](README.md)

# Video

## Text to video generation

- [ModelScope Text to video synthesis](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis)
  - [zeroscope v2 xl](https://huggingface.co/cerspense/zeroscope_v2_XL) Watermark free modelscope based video model generating high quality video at 1024x576 16:9, to be used with [text2video extension](https://github.com/kabachuha/sd-webui-text2video) for automatic1111
- [Nvidia VideoLDM: Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)
- [Potat1](https://huggingface.co/camenduru/potat1) , [colab](https://github.com/camenduru/text-to-video-synthesis-colab)
- [Phenaki](https://openreview.net/forum?id=vOEXS39nOF) multi minute text to video prompts with scene changes, [project page](https://phenaki.video/)
- [StableVideo](https://rese1f.github.io/StableVideo/) Text-driven Consistency-aware Diffusion Video Editing, [code](https://github.com/rese1f/StableVideo), [paper](https://rese1f.github.io/StableVideo/)
- [Rerender A Video](https://github.com/williamyang1991/Rerender_A_Video) Zero-Shot Text-Guided Video-to-Video Translation, [paper](https://arxiv.org/abs/2306.07954)
- [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter) Open Diffusion Models for High-Quality Video Generation
- [i2vgen-xl](https://github.com/ali-vilab/i2vgen-xl) a holistic video generation ecosystem for video generation building on diffusion models
- [pixeldance](https://makepixelsdance.github.io/) High-Dynamic Video Generation
- [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) aims to reproduce Sora
- [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) Consistent Long-Range Image and Video Generation
- [Open-Sora](https://github.com/hpcaitech/Open-Sora) Open implementation approach for video generation
- [CogVideo](https://github.com/THUDM/CogVideo) SOTA video generation and consistency generating 6 seconds of video with 8fps at 720x480 using 18-36GB vRAM

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
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) Tracking Anything with Decoupled Video Segmentation, [paper](https://arxiv.org/abs/2309.03903)
- [Cutie](https://github.com/hkchengrex/Cutie) Putting the Object Back into Video Object Segmentation, [paper](https://arxiv.org/abs/2310.12982)
- [YOLOv10](https://github.com/THU-MIG/yolov10) Real-Time End-to-End Object Detection
- [SAM2](https://github.com/facebookresearch/segment-anything-2) enables fast, precise selection of any object in any video or image

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
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) reference implementation of "3D Gaussian Splatting for Real-Time Radiance Field Rendering", [paper](https://arxiv.org/abs/2308.04079v1)
- [4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting, [paper](https://arxiv.org/abs/2310.10642v1)

## Deepfakes

- [roop](https://github.com/s0md3v/roop) one-click deepfake (face swap)
  - [rope](https://github.com/Hillobar/Rope) GUI-focused roop
- [streamv2v](https://github.com/Jeff-LiangF/streamv2v) Official Pytorch implementation of [StreamV2V](https://jeff-liangf.github.io/projects/streamv2v/)
- [MusePose](https://github.com/TMElyralab/MusePose) Pose Driven Image 2 Video framework to generate Virtual Humans 
- [V-Express](https://github.com/tencent-ailab/V-Express) generate a talking head video under the control of a reference image, an audio, and a sequence of V-Kps images
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) real time face swap and one-click video deepfake with only a single image

## Benchmarking

- [MSU Benchmarks](https://videoprocessing.ai/) collection of video processing benchmarks developed by the Video Processing Group at the Moscow State University
- [Video Super Resolution Benchmarks](https://paperswithcode.com/task/video-super-resolution)
- [Video Generation Benchmarks](https://paperswithcode.com/task/video-generation)
- [Video Frame Interpolation Benchmarks](https://paperswithcode.com/task/video-frame-interpolation)

## Inpainting Outpainting

- [ProPainter](https://github.com/sczhou/ProPainter) Improving Propagation and Transformer for Video Inpainting, [paper](https://arxiv.org/abs/2309.03897)
