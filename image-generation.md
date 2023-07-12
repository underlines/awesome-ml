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

text to 3d:
- [OpenAI shap-E](https://github.com/openai/shap-e) a text/image to 3D model
- [shap-e local](https://github.com/kedzkiest/shap-e-local) run text-to-3d locally
- [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) A PyTorch implementation of the text-to-3D model Dreamfusion using the Stable Diffusion text-to-2D model

other:
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) image segmentation
  - [YOLOv8](https://github.com/ultralytics/ultralytics) SOTA object detection, segmentation, classification and tracking
  - [DINOv2](https://github.com/facebookresearch/dinov2) 1B-parameter ViT model to generate robust all-purpose visual features that outperform OpenCLIP benchmarks at image and pixel levels
- [Final2x](https://github.com/Tohrusky/Final2x) Image super-resolution through interpolation supporting multiple models like RealCUGAN, ESRGAN, Waifu2x, SRMD
-  [text-to-room](https://lukashoel.github.io/text-to-room/) text to room
- [DragGAN](https://github.com/XingangPan/DragGAN) Interactive Point-based Manipulation on Generative Images, [demo](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)

## Wrappers & GUIs
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) powerful and modular stable diffusion pipelines using a graph/nodes/flowchart based interface, runs SDXL 0.9, SD2.1, SD2.0, SD1.5
  - [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) installs missing custom nodes automatically
  - [SeargeSDXL](https://github.com/SeargeDP/SeargeSDXL) Custom SDXL Node for easier SDXL usage and img2img workflow that utilizes base & refiner
  - [Sytan ComfyUI SDXL workflow](https://github.com/SytanSD/Sytan-SDXL-ComfyUI/tree/main) with txt2img using base and refiner
- [Automatic1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) well known UI for Stable Diffusion
  - [sd-webui-cloud-inference](https://github.com/omniinfer/sd-webui-cloud-inference) extension via omniinfer.io
- [SD.Next](https://github.com/vladmandic/automatic) vladmandic/automatic Fork, seemingly more active development efforts compared to automatic1111's original repo
- [stable-diffusion-xl-demo](https://github.com/FurkanGozukara/stable-diffusion-xl-demo) runs SDXL 0.9 in a basic interface
- [imaginAIry](https://github.com/brycedrennan/imaginAIry/blob/master/README.md) a Stable Diffusion UI
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)  Alternative, polished stable diffusion UI with less features than automatic1111
- [mlc-ai/web-stable-diffusion](https://github.com/mlc-ai/web-stable-diffusion)
- [anapnoe/stable-diffusion-webui-ux](https://github.com/anapnoe/stable-diffusion-webui-ux) Redesigned from automatic1111's UI, adding mobile and desktop layouts and UX improvements
- [refacer](https://github.com/xaviviro/refacer) One-Click Deepfake Multi-Face Swap Tool


## Fine Tuning
- https://github.com/JoePenna/Dreambooth-Stable-Diffusion
- https://github.com/TheLastBen/fast-stable-diffusion
- https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- https://github.com/cloneofsimo/lora
- [StableTuner](https://github.com/devilismyfriend/StableTuner) Windows GUI for Finetuning / Dreambooth Stable Diffusion models
- [sd-scripts](https://github.com/kohya-ss/sd-scripts) by kohya-ss
  - [LoRA Easy Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts) GUI for Kohya's Scripts
  - [Kohya_ss](https://github.com/bmaltais/kohya_ss) Windows-focused Gradio GUI for Kohya's Stable Diffusion trainers, [experimental](https://github.com/bmaltais/kohya_ss/tree/sdxl) sdxl support, [reddit thread](https://www.reddit.com/r/StableDiffusion/comments/14xhpxm/dreambooth_sdxl_09/)
- [Fine tuning concepts explained visually](https://github.com/cloneofsimo/lora/discussions/67)
- [text2image-gui](https://github.com/n00mkrad/text2image-gui) a Stable Diffusion GUI by NMKD

## Research
 - [Speed Is All You Need](https://arxiv.org/abs/2304.11267) up to 50% speed increase for Latent Diffusion Models
 - [ORCa](https://arxiv.org/abs/2212.04531) converts glossy objects into radiance-field cameras, enabling depth estimation and novel-view synthesis, [project](https://ktiwary2.github.io/objectsascam/), [code](https://github.com/ktiwary2/orca)
 - [cocktail](https://mhh0318.github.io/cocktail/) Mixing Multi-Modality Controls for Text-Conditional Image Generation, [project](https://mhh0318.github.io/cocktail/), [code](https://github.com/mhh0318/Cocktail)
 - [SnapFusion](https://snap-research.github.io/SnapFusion/) Fast text-to-image diffusion on mobile phones in 2 seconds
 - [Objaverse-xl](https://objaverse.allenai.org/objaverse-xl-paper.pdf) dataset of 10 million annotated high quality 3D objects, [hf](https://huggingface.co/datasets/allenai/objaverse)
