# Large Language Models
- https://github.com/borzunov/chat.petals.ml / http://chat.petals.ml/
- https://github.com/LAION-AI/Open-Assistant / https://projects.laion.ai/Open-Assistant/ / https://open-assistant.io
- https://github.com/microsoft/visual-chatgpt

## LLaMA models
| Model  | Size | Quantization | Fine-Tuning | ggml .bin                                                                               | hf .pt                                                                                                                                                                                                                        | native .bin                                                                                                                                                                                                                                                                                                                           |
| ------ | ---- | ------------ | ----------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| alpaca | 7b   | 4bit GPTQ    | native      | [Sosaka/Alpaca-native-4bit-ggml](https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml) | [ozcur/alpaca-native-4bit](https://huggingface.co/ozcur/alpaca-native-4bit)                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 7b   | 8bit         | native      |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 7b   | 16bit        | native      |                                                                                         |                                                                                                                                                                                                                               | [chavinlo/alpaca-native](https://huggingface.co/chavinlo/alpaca-native)                                                                                                                                                                                                                                                               |
| alpaca | 13b  | 4bit GPTQ    | native      |                                                                                         |                                                                                                                                                                                                                               | [Dogge/alpaca-13b](https://huggingface.co/Dogge/alpaca-13b)                                                                                                                                                                                                                                                                           |
| alpaca | 13b  | 8bit         | native      |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 30b  | 4bit GPTQ    | native      | [Pi3141/alpaca-30B-ggml](https://huggingface.co/Pi3141/alpaca-30B-ggml)                 |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 30b  | 8bit         | native      |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 7b   | 4bit GPTQ    | LoRa        | <br>[Pi3141/alpaca-7B-ggml](https://huggingface.co/Pi3141/alpaca-7B-ggml)               |                                                                                                                                                                                                                               | [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b/)                                                                                                                                                                                                                                                                  |
| alpaca | 7b   | 8bit         | LoRa        |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 7b   | 16bit        | LoRa        |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 13b  | 4bit GPTQ    | LoRa        | [Pi3141/alpaca-13B-ggml](https://huggingface.co/Pi3141/alpaca-13B-ggml)                 | [elinas/alpaca-13b-lora-int4](https://huggingface.co/elinas/alpaca-13b-lora-int4)                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 13b  | 8bit         | LoRa        |                                                                                         |                                                                                                                                                                                                                               | [baruga/alpaca-lora-13b](https://huggingface.co/baruga/alpaca-lora-13b)<br>[chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b)<br>[Draff/llama-alpaca-stuff](https://huggingface.co/Draff/llama-alpaca-stuff/tree/main/Alpaca-Loras)<br>[samwit/alpaca13B-lora](https://huggingface.co/samwit/alpaca13B-lora) |
| alpaca | 30b  | 4bit GPTQ    | LoRa        |                                                                                         | [elinas/alpaca-30b-lora-int4](https://huggingface.co/elinas/alpaca-30b-lora-int4)                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 30b  | 8bit         | LoRa        |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| alpaca | 30b  | 16bit        | LoRa        |                                                                                         |                                                                                                                                                                                                                               | [baseten/alpaca-30b](https://huggingface.co/baseten/alpaca-30b)                                                                                                                                                                                                                                                                       |
|        |      |              |             |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 7b   | 4bit GPTQ    | none        | [Pi3141/llama-7B-ggml](https://huggingface.co/Pi3141/llama-7B-ggml)                     |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 7b   | 8bit         | none        |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 7b   | 16bit        | none        |                                                                                         | [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)                                                                                                                                         |
| llama  | 13b  | 4bit GPTQ    | none        | [Pi3141/llama-13B-ggml](https://huggingface.co/Pi3141/llama-13B-ggml)                   |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 13b  | 8bit         | none        |                                                                                         |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 30b  | 4bit GPTQ    | none        | [Pi3141/llama-30B-ggml](https://huggingface.co/Pi3141/llama-30B-ggml)                   | [kuleshov/llama-30b-4bit](https://huggingface.co/kuleshov/llama-30b-4bit)<br>[elinas/llama-30b-int4](https://huggingface.co/elinas/llama-30b-int4)<br>[TianXxx/llama-30b-int4](https://huggingface.co/TianXxx/llama-30b-int4) |                                                                                                                                                                                                                                                                                                                                       |
| llama  | 65b  | 4bit GPTQ    | none        | [Pi3141/llama-65B-ggml](https://huggingface.co/Pi3141/llama-65B-ggml)                   |                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                       |


## Data sets
- [Alpaca-lora instruction finetuned using Low Rank Adaption](https://github.com/tloen/alpaca-lora)
- [codealpaca Instruction training data set for code generation](https://github.com/sahil280114/codealpaca)

## Research
- [LLM Model Cards](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs)
- [GPTs are GPTs: An early look at the labor market impact potential of LLMs](https://arxiv.org/abs/2303.10130)
- [ViperGPT Visual Inference via Python Execution for reasoning](https://viper.cs.columbia.edu/)
- [Emergent Abilities of LLMs ](https://openreview.net/forum?id=yzkSU5zdwD), [blog post](https://www.jasonwei.net/blog/emergence)

## LLM GUIs
### OpenAI
- [chatgptui/desktop](https://github.com/chatgptui/desktop)
- [TypingMind](https://www.typingmind.com/)
- [Chatwithme.chat](https://www.chatwithme.chat/)
- [datafilik/GPT-Voice-Assistant](https://github.com/datafilik/GPT-Voice-Assistant)
- [Abdallah-Ragab/VoiceGPT](https://github.com/Abdallah-Ragab/VoiceGPT)
- [LlmKira/Openaibot](https://github.com/LlmKira/Openaibot)
- [chathub-dev/chathub](https://github.com/chathub-dev/chathub)
- [enricoros/nextjs-chatgpt-app](https://github.com/enricoros/nextjs-chatgpt-app)
- [no8081/chatgpt-demo](https://github.com/ddiu8081/chatgpt-demo)

### Others
- [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
- [Text Generation Webui | An all purpose UI to run LLMs of all sorts with optimizations](https://github.com/oobabooga/text-generation-webui) ([running LLaMA on less than 10GB vram](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134), [running LLaMA-7b on a 3080](https://github.com/TimDettmers/bitsandbytes/issues/30#issuecomment-1455993902), [detailed guide](https://rentry.org/llama-tard-v2))
- [Alpaca-LoRa-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)
- [ChatLLaMA | another implementation](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Dalai](https://github.com/cocktailpeanut/dalai)
- [ChatLLaMA | LLaMA-based ChatGPT for single GPUs](https://github.com/juncongmoo/chatllama)

## LLM Wrappers
- [acheong08/ChatGPT Python](https://github.com/acheong08/ChatGPT)
- [mpoon/gpt-repository-loader](https://github.com/mpoon/gpt-repository-loader)
- [LangChain | framework for developing LLM applications](https://github.com/hwchase17/langchain) ([example](https://www.youtube.com/watch?v=iRJ4uab_NIg&t=588s))
- [LangFlow | GUI for Langchain](https://github.com/logspace-ai/langflow)
- [pyllama | hacked version of LLaMA based on Meta's implementation, optimized for Single GPUs](https://github.com/juncongmoo/pyllama)
- [Toolformer implementation | Allows LLMs to use Tools](https://github.com/lucidrains/toolformer-pytorch)
- [FastLLaMA Python wrapper for llama.cpp](https://github.com/PotatoSpudowski/fastLLaMa)

## Showcases
- [Opinionate.io AI Debating AI](https://opinionate.io/)
- 

## Fine Tuning
- [simple llama finetuner](https://github.com/lxe/simple-llama-finetuner)
- 

# Image Generation

## Models
- https://github.com/kakaobrain/karlo
- https://lukashoel.github.io/text-to-room/

## Wrappers & GUIs
- https://github.com/brycedrennan/imaginAIry/blob/master/README.md
- https://github.com/invoke-ai/InvokeAI
- https://github.com/AUTOMATIC1111/stable-diffusion-webui
- [mlc-ai/web-stable-diffusion](https://github.com/mlc-ai/web-stable-diffusion)

## Fine Tuning
- https://github.com/JoePenna/Dreambooth-Stable-Diffusion
- https://github.com/TheLastBen/fast-stable-diffusion
- https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- https://github.com/cloneofsimo/lora

# Benchmarking
- https://videoprocessing.ai/benchmarks/
- https://paperswithcode.com/

# Video
## Text to video generation
- [ModelScope Text to video synthesis](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis)

## Frame Interpolation (Temporal Interpolation)
- https://github.com/google-research/frame-interpolation
- https://github.com/ltkong218/ifrnet
- https://github.com/megvii-research/ECCV2022-RIFE
- 

## Super Resolution (Spacial Interpolation)
- https://github.com/researchmm/FTVSR
- https://github.com/picsart-ai-research/videoinr-continuous-space-time-super-resolution
- 

## Spacio Temporal Interpolation
- https://github.com/llmpass/RSTT

# Audio
## Compression
- https://github.com/facebookresearch/encodec

## Speech Recognition
- https://github.com/openai/whisper
- 

# AI DevOps
- https://www.steamship.com/

# Optimization
## Inference
- https://github.com/bigscience-workshop/petals
- https://github.com/chavinlo/distributed-diffusion
- https://github.com/VoltaML/voltaML-fast-stable-diffusion
- https://github.com/FMInference/FlexGen
- https://github.com/alpa-projects/alpa
- https://github.com/kir-gadjello/zipslicer
- https://github.com/modular-ml/wrapyfi-examples_llama
- https://github.com/tloen/llama-int8
- [4 bits quantization of LLaMa using GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa) ([discussion](https://github.com/oobabooga/text-generation-webui/issues/177))

## Training
- https://github.com/learning-at-home/hivemind

## Other Optimization
- https://github.com/HazyResearch/flash-attention
- https://github.com/stochasticai/x-stable-diffusion
- 
