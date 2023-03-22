# How to run LLaMA 4bit models on Windows

## Table of contents

- [Windows 11 WSL2 Ubuntu / Native Ubuntu](#windows-11-wsl2-ubuntu--native-ubuntu)
- [Windows 11 native](#windows-11-native)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

# Windows 11 WSL2 Ubuntu / Native Ubuntu

## Install Ubuntu WSL2 on Windows 11
1. Install WSL2 on Windows Store
2. Install Ubuntu
3. Start Windows Terminal in Ubuntu

## Install Anaconda + Build Essentials
1. `sudo apt update`
2. `sudo apt upgrade`
3. `mkdir Downloads`
4. `cd Downloads/`
5. `wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh`
6. `chmod +x ./Anaconda3-2022.05-Linux-x86_64.sh`
7. `./Anaconda3-2022.05-Linux-x86_64.sh`
8. `sudo apt install build-essential`
9. `cd ..`

## Install text-generation-webui
text-generation-webui install [instructions](https://github.com/oobabooga/text-generation-webui):
1. `conda create -n textgen`
2. `conda activate textgen`
3. `conda install torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia`
4. `git clone https://github.com/oobabooga/text-generation-webui`
5. `cd text-generation-webui`
6. `pip install -r requirements.txt`

## Build GPTQ for LLaMA to enable 4bit support
4-bit installation [instructions](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#4-bit-mode):
1. `pip uninstall transformers`
2. `pip install git+https://github.com/zphang/transformers@llama_push`
3. `mkdir repositories`
4. `cd repositories`
5. `git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa`
6. `cd GPTQ-for-LLaMa`
7. `python setup_cuda.py install`

## Download the model files
Download the tokenizer and config files for the model size you want and change the size in the command below accordingly: 7b / 13b / 30b / 65b
1. `cd ../../`
2. `python download-model.py --text-only decapoda-research/llama-13b-hf`

Download the 4bit model itself from [this](https://huggingface.co/decapoda-research/llama-13b-hf-int4/tree/main) repo, change the url accordingly: 7b / 13b / 30b / 65b
1. `explorer.exe .` to open Windows Explorer showing the Ubuntu folder. (windows only)
2. Move the llama-13b-4bit.pt file into `/text-generation-webui/models/` (not in the subfolder llama-13b-hf)

The folder structure should be:
```
\text-generation-webui\models\llama-13b-4bit.pt
\text-generation-webui\models\llama-13b-hf\config.json
\text-generation-webui\models\llama-13b-hf\generation_config.json
\text-generation-webui\models\llama-13b-hf\pytorch_model.bin.index.json
\text-generation-webui\models\llama-13b-hf\special_tokens_map.json
\text-generation-webui\models\llama-13b-hf\tokenizer.model
\text-generation-webui\models\llama-13b-hf\tokenizer_config.json
```

## Run
Various ways to run LLaMA in text-generation-webui:
1. `python server.py --model llama-13b-hf --load-in-4bit --no-stream` if generation becomes very slow after some time, due to [issue](https://github.com/oobabooga/text-generation-webui/issues/147) in 4 bit mode, turn off streaming
2. `python server.py --model llama-13b-hf --load-in-4bit` if there are not slow down issues
3. `python server.py --model llama-13b-hf --load-in-4bit --no-stream --chat` starting in chat mode, also possible both with or without --no-stream 

# Windows 11 native

## Install Miniconda
1. Download and install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
2. Open `Anaconda Prompt (Miniconda 3)` from the Start Menu

## Install text-generation-webui
text-generation-webui install [instructions](https://github.com/oobabooga/text-generation-webui):
1. In the Anaconda Prompt run: `conda create -n textgen`
2. `conda activate textgen`
3. `conda install torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia`
4. `git clone https://github.com/oobabooga/text-generation-webui`
5. `cd text-generation-webui`
6. `pip install -r requirements.txt`

## Install GPTQ for LLaMA to enable 4bit support
Alternatively, you can use [prebuilt GPTQ Wheels for Windows](https://github.com/oobabooga/text-generation-webui/issues/177#issuecomment-1464844721):
1. `pip uninstall transformers`
2. `pip install git+https://github.com/zphang/transformers@llama_push`
3. `mkdir repositories`
4. `cd repositories`
5. `git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa`
6. `cd GPTQ-for-LLaMa`
7. Download the [prebuilt wheels](https://github.com/oobabooga/text-generation-webui/files/10947842/quant_cuda-0.0.0-cp310-cp310-win_amd64.whl.zip) and unzip in the same directory C:\Users\yourname\text-generation-webui\repositories\GPTQ-for-LLaMa\
8. `python -m pip install quant_cuda-0.0.0-cp310-cp310-win_amd64.whl`
9. Don't run python setup_cuda install.

Download the tokenizer and config files for the model size you want and change the size in the command below accordingly: 7b / 13b / 30b / 65b
1. `cd ..\..\`
2. `python download-model.py --text-only decapoda-research/llama-13b-hf`

Download the 4bit model itself from [this](https://huggingface.co/decapoda-research/llama-13b-hf-int4/tree/main) repo, change the url accordingly: 7b / 13b / 30b / 65b
1. `explorer.exe .` to open Windows Explorer showing the current folder.
2. Move the llama-13b-4bit.pt file into `/text-generation-webui/models/` (not in the subfolder llama-13b-hf)

The folder structure should be:
```
\text-generation-webui\models\llama-13b-4bit.pt
\text-generation-webui\models\llama-13b-hf\config.json
\text-generation-webui\models\llama-13b-hf\generation_config.json
\text-generation-webui\models\llama-13b-hf\pytorch_model.bin.index.json
\text-generation-webui\models\llama-13b-hf\special_tokens_map.json
\text-generation-webui\models\llama-13b-hf\tokenizer.model
\text-generation-webui\models\llama-13b-hf\tokenizer_config.json
```

## Run
Various ways to run LLaMA in text-generation-webui:
1. `python server.py --model llama-13b-hf --load-in-4bit --no-stream` if generation becomes very slow after some time, due to [issue](https://github.com/oobabooga/text-generation-webui/issues/147) in 4 bit mode, turn off streaming
2. `python server.py --model llama-13b-hf --load-in-4bit` if there are not slow down issues
3. `python server.py --model llama-13b-hf --load-in-4bit --no-stream --chat` starting in chat mode, also possible both with or without --no-stream 

# Troubleshooting

- [text-generation-webui general LLaMA support](https://github.com/oobabooga/text-generation-webui/issues/147)
- [text-generation-webui GPTQ 4bit support for LLaMA issues](https://github.com/oobabooga/text-generation-webui/issues/177)
- [text-generation-webui GPTQ 4bit support for LLaMA pull request](https://github.com/oobabooga/text-generation-webui/pull/206)
- [4channel thread, might contain NSFW discussions, but has some useful info](https://boards.4channel.org/vg/thread/421001187/aids-ai-dynamic-storytelling-general)

If you don't want to use WSL2 Ubuntu, but run natively on windows, these resources might be helpful:

- [Windows Wheels to compile GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/11#issuecomment-1464958666) for quant_cuda in order to install GPTQ-for-LLaMA
- Prebuilt [Windows Binaries](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134) for bitsandbytes with cuda support (only step 5-6)

If you're looking for Apple Silicon support:

- https://news.ycombinator.com/item?id=35100086

# Resources

## converted models

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

## alpaca
[alpaca from stanford university](https://crfm.stanford.edu/2023/03/13/alpaca.html) is an instruction fine tuned llama model. They opensourced the fine tuning resources including the dataset. People started to replicate this but using [LoRa fine tuning](https://github.com/tloen/alpaca-lora) with llama-7b and even llama-13b:

Note that some people fine-tuned on the original alpaca dataset, which has many low quality examples, while people on [this issue](https://github.com/tloen/alpaca-lora/pull/32) made a huge effort cleaning and improving the original alpaca dataset for better RLHF:

## text-generation-webui settings presets:
1. `explorer.exe .` to open the Ubuntu path to the text-generation-webui in Windows Explorer
2. navigate to `/presets` and make a copy of `NovelAI-Sphinx Moth.txt` and name it for example `llama-13b-4bit.txt`
3. Edit the settings according to some examples below

[Szpadel @ HN](https://news.ycombinator.com/item?id=35101869)
```
do_sample=True
top_p=0.9
top_k=30
temperature=0.62
repetition_penalty=1.08
typical_p=1.0
```

## Other guides
- https://rentry.org/llama-tard-v2#bonus-4-4bit-llama-basic-setup
- Nerdy Rodent's excellent [Video Tutorial](https://www.youtube.com/watch?v=rGsnkkzV2_o), but diverges from this guide

## Other tools
- https://github.com/hwchase17/langchain ([example](https://www.youtube.com/watch?v=iRJ4uab_NIg&t=588s))
- [ChatLLaMA 📢 Open source implementation for LLaMA-based ChatGPT runnable in a single GPU. 15x faster training process than ChatGPT](https://github.com/juncongmoo/chatllama)
- [ChatLLaMA another implementation](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)
- [pyllama: a hacked version of LLaMA based on original Facebook's implementation but more convenient to run in a Single consumer grade GPU.](https://github.com/juncongmoo/pyllama)
- [alpaca-lora: reproducing the Stanford Alpaca InstructLLaMA result on consumer hardware](https://github.com/tloen/alpaca-lora)
- [Implementation of Toolformer, Language Models That Can Use Tools, by MetaAI](https://github.com/lucidrains/toolformer-pytorch)



