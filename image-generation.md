[🏠Home](README.md)

# Image Generation

## Models

 text2image:

- [karlo](https://github.com/kakaobrain/karlo) text2image model
- [DeepFloyd if by StabilityAI](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) open-source text-to-image model with photorealism and language understanding. [code](https://github.com/deep-floyd/IF)
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) multilingual text2image latent diffusion model
- [stable diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stable diffusion 2.0](https://huggingface.co/stabilityai/stable-diffusion-2)
- [stable diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- stable diffusion xl (SDXL) [base 0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) & [refinder 0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff) Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning
- [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis, [paper](https://arxiv.org/abs/2310.00426)
- [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model) LoRAs for high quality few step image generation
- [OnnxStream](https://github.com/vitoplantamura/OnnxStream) Stable Diffusion XL 1.0 Base with 298MB of RAM
- [StreamDiffusion](https://github.com/cumulo-autumn/streamdiffusion) A Pipeline-Level Solution for Real-Time Interactive Generation
- [AnyText](https://github.com/tyxsspa/AnyText) Code and Model for a diffusion pipeline covering a latent module and text embedding to generate and manipulate text in images
- [InstantID](https://github.com/InstantID/InstantID) Zero-shot Identity-Preserving Generation in Seconds, [ComfyUI plugin](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker) Rapid customization within seconds, with no additional LoRA training preserving ID with high fidelity and text controllability which can serve as an adapter for other models
- [StableCascade](https://github.com/Stability-AI/StableCascade) successor to Stable Diffusion by Stability AI with smaller latent space, higher speeds and better quality
- [IDM-VTON](https://github.com/yisol/IDM-VTON) Virtual Try-on for clothes and fashion
- [ConsistentID](https://github.com/JackAILab/ConsistentID) Portrait Generation with Multimodal Fine-Grained Identity Preservation
- [Flux](https://huggingface.co/black-forest-labs) Black Forrest Labs consisting of ex stabilityAi staff built a SOTA text-to-image model Flux and Flux schnell, a 13B parameter transformer capable of writing text, following complex prompts released under apache 2 license
- [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) multimodal autoregressive LLMs capable of generating flexible and photorealistic images from text descriptions 

 text to 3d:

- [OpenAI shap-E](https://github.com/openai/shap-e) a text/image to 3D model
- [shap-e local](https://github.com/kedzkiest/shap-e-local) run text-to-3d locally
- [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) A PyTorch implementation of the text-to-3D model Dreamfusion using the Stable Diffusion text-to-2D model

 image to 3d:

- [Wonder3D](https://github.com/xxlong0/Wonder3D) A cross-domain diffusion model for 3D reconstruction from a single image
- [DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D) Official implementation of DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior

 image to text (OCR):

- [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) LaTeX OCR

other:

- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) image segmentation
  - [YOLOv8](https://github.com/ultralytics/ultralytics) SOTA object detection, segmentation, classification and tracking
  - [DINOv2](https://github.com/facebookresearch/dinov2) 1B-parameter ViT model to generate robust all-purpose visual features that outperform OpenCLIP benchmarks at image and pixel levels
  - [segment-anything-fast](https://github.com/pytorch-labs/segment-anything-fast) A batched offline inference oriented version of segment-anything
- [Final2x](https://github.com/Tohrusky/Final2x) Image super-resolution through interpolation supporting multiple models like RealCUGAN, ESRGAN, Waifu2x, SRMD
- [text-to-room](https://lukashoel.github.io/text-to-room/) text to room
- [DragGAN](https://github.com/XingangPan/DragGAN) Interactive Point-based Manipulation on Generative Images, [demo](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
- [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion) Harnessing Diffusion Models for Interactive Point-based Image Editing
- [HQTrack](https://github.com/jiawen-zhu/hqtrack) Tracking Anything in High Quality (HQTrack) is a framework for high performance video object tracking and segmentation
- [CoTracker](https://github.com/facebookresearch/co-tracker) It is Better to Track Together. A fast transformer-based model that can track any point in a video
- [ZeroNVS](https://arxiv.org/pdf/2310.17994.pdf) Zero shot 460 degree view synthesis from single images
- [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion) Real-time inference for Stable Diffusion - 0.88s latency
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) Better depth estimation including a ControlNet for ComfyUI and ONNX and TensorRT versions
- [SUPIR](https://github.com/Fanghua-Yu/SUPIR) Super Resolution and Image Restoration
- [RMBG](https://huggingface.co/briaai/RMBG-1.4) BRIA Background Removal model. [hf demo space](https://huggingface.co/spaces/briaai/BRIA-RMBG-1.4)

## Wrappers & GUIs

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) powerful and modular stable diffusion pipelines using a graph/nodes/flowchart based interface, runs SDXL 0.9, SD2.1, SD2.0, SD1.5
  - [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) installs missing custom nodes automatically
  - [SeargeSDXL](https://github.com/SeargeDP/SeargeSDXL) Custom SDXL Node for easier SDXL usage and img2img workflow that utilizes base & refiner
  - [Sytan ComfyUI SDXL workflow](https://github.com/SytanSD/Sytan-SDXL-ComfyUI/tree/main) with txt2img using base and refiner
- [Automatic1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) well known UI for Stable Diffusion
  - [sd-webui-cloud-inference](https://github.com/omniinfer/sd-webui-cloud-inference) extension via omniinfer.io
  - [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) platform on top of SDWebUI to make development easier, optimize resource management, and speed up inference
- [SD.Next](https://github.com/vladmandic/automatic) vladmandic/automatic Fork, seemingly more active development efforts compared to automatic1111's original repo
- [Fooocus](https://github.com/lllyasviel/Fooocus) Midjourney alike GUI for SDXL to focus on prompting and generating
  - [RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) A Fooocus fork
  - [Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) A Fooocus fork
- [stable-diffusion-xl-demo](https://github.com/FurkanGozukara/stable-diffusion-xl-demo) runs SDXL 0.9 in a basic interface
- [imaginAIry](https://github.com/brycedrennan/imaginAIry/blob/master/README.md) a Stable Diffusion UI
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)  Alternative, polished stable diffusion UI with less features than automatic1111
- [mlc-ai/web-stable-diffusion](https://github.com/mlc-ai/web-stable-diffusion)
- [anapnoe/stable-diffusion-webui-ux](https://github.com/anapnoe/stable-diffusion-webui-ux) Redesigned from automatic1111's UI, adding mobile and desktop layouts and UX improvements
- [refacer](https://github.com/xaviviro/refacer) One-Click Deepfake Multi-Face Swap Tool
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) CPU inference of Stable Diffusion in pure C/C++ with huge performance gains, supporting ggml, 16/32 bit float, 4/5/8 bit quantization, AVX/AVX2/AVX512, SD1.x, SD2.x, txt2img/img2img
- [FaceFusion](https://github.com/facefusion/facefusion) Next generation face swapper and enhancer
- [OneFlow](https://github.com/Oneflow-Inc/diffusers) Backend for diffusers and ComfyUI

## Fine Tuning

- https://github.com/JoePenna/Dreambooth-Stable-Diffusion
- [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion) TheLastBen's Repo for SD, SDXL fine-tuning and DreamBooth on RunPod, Paperspace, Colab and others
- https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- https://github.com/cloneofsimo/lora
- [OneTrainer](https://github.com/Nerogar/OneTrainer) all in one training for SD, SDXL and inpainting models supporting fine-tuning, LoRA, embeddings
- [sd-scripts](https://github.com/kohya-ss/sd-scripts) by kohya-ss
  - [LoRA Easy Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts) GUI for Kohya's Scripts
  - [Kohya_ss](https://github.com/bmaltais/kohya_ss) Windows-focused Gradio GUI for Kohya's Stable Diffusion trainers, [experimental](https://github.com/bmaltais/kohya_ss/tree/sdxl) sdxl support, [reddit thread](https://www.reddit.com/r/StableDiffusion/comments/14xhpxm/dreambooth_sdxl_09/)
- [Fine tuning concepts explained visually](https://github.com/cloneofsimo/lora/discussions/67)
- [text2image-gui](https://github.com/n00mkrad/text2image-gui) a Stable Diffusion GUI by NMKD
- [sd-webui-EasyPhoto](https://github.com/aigc-apps/sd-webui-EasyPhoto) / [easyphoto](https://github.com/aigc-apps/easyphoto) plugin for generating AI portraits that can be used to train digital doppelgangers with 5-10 photos and a quick LoRA fine tune, [paper](https://arxiv.org/abs/2310.04672v1)
- [StableTuner](https://github.com/devilismyfriend/StableTuner) Windows GUI for Finetuning / Dreambooth Stable Diffusion models (abandoned)
- [SimpleTuner](https://github.com/bghira/SimpleTuner) fine-tuning for StableDiffusion, PixArt, Flux with LoRA and full U-Net training, multi GPU support, DeepSpeed
- [x-flux](https://github.com/XLabs-AI/x-flux) LoRA and ControlNet training scripts for Flux model by Black Forest Labs using DeepSpeed

## Research

- [Speed Is All You Need](https://arxiv.org/abs/2304.11267) up to 50% speed increase for Latent Diffusion Models
- [ORCa](https://arxiv.org/abs/2212.04531) converts glossy objects into radiance-field cameras, enabling depth estimation and novel-view synthesis, [project](https://ktiwary2.github.io/objectsascam/), [code](https://github.com/ktiwary2/orca)
- [cocktail](https://mhh0318.github.io/cocktail/) Mixing Multi-Modality Controls for Text-Conditional Image Generation, [project](https://mhh0318.github.io/cocktail/), [code](https://github.com/mhh0318/Cocktail)
- [SnapFusion](https://snap-research.github.io/SnapFusion/) Fast text-to-image diffusion on mobile phones in 2 seconds
- [Objaverse-xl](https://objaverse.allenai.org/objaverse-xl-paper.pdf) dataset of 10 million annotated high quality 3D objects, [hf](https://huggingface.co/datasets/allenai/objaverse)
- [LightGlue](https://github.com/cvg/LightGlue) Local Feature Matching at Light Speed, a lightweight feature matcher with high accuracy and blazing fast inference. It takes as input a set of keypoints and descriptors for each image and returns the indices of corresponding points
- [ml-mgie](https://github.com/apple/ml-mgie) Guiding Instruction-based Image Editing via Multimodal Large Language Models
- [VAR](https://github.com/FoundationVision/VAR) GPT beats diffusion
- [InstantStyle](https://github.com/InstantStyle/InstantStyle) towards Style-Preserving in Text-to-Image Generation
- 
