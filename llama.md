#
Guide: https://rentry.org/llama-tard-v2#bonus-4-4bit-llama-basic-setup

# Installation Windows 11 WSL2 Ubuntu + Anaconda + build-essentials
1. Install WSL2 on Windows Store
2. Install Ubuntu
3. Start Windows Terminal in Ubuntu
4. Install Anaconda3
5. `sudo apt install build-essential`

Follow text-generation-webui install [instructions](https://github.com/oobabooga/text-generation-webui)
1. `conda create -n textgen`
2. `conda activate textgen`
3. `conda install torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia`
4. `git clone https://github.com/oobabooga/text-generation-webui`
5. `cd text-generation-webui`
6. `pip install -r requirements.txt`

Follow the 4-bit installation [instructions](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#4-bit-mode)
1. `pip uninstall transformers`
2. `pip install git+https://github.com/zphang/transformers@llama_push`
3. `mkdir repositories`
4. `cd repositories`
5. `git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa`
6. `cd GPTQ-for-LLaMa`
7. `python setup_cuda.py install`

Download the tokenizer and config files for the model size you want and change the size in the command below accordingly: 7b / 13b / 30b / 65b
1. `python download-model.py --text-only decapoda-research/llama-13b-hf`

Download the 4bit model itself from [this](https://huggingface.co/decapoda-research/llama-13b-hf-int4/tree/main) repo, change the url accordingly: 7b / 13b / 30b / 65b
1. In the Windows Terminal on Ubuntu enter `explorer.exe .` to open Windows Explorer showing the Ubuntu folder.
2. Move the llama-13b-4bit.pt file into `/text-generation-webui/models/` (not in the subfolder llama-13b-hf)

Start (disable streaming, due to a memory problem in 4 bit mode:
2. `python server.py --model llama-13b-hf --load-in-4bit --no-stream`
