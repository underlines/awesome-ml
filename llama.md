# Text-generation-webui manual installation on Windows WSL2 Ubuntu / Ubuntu Native / Windows Native

For a simple automatic install, use the one-click installers provided on
The video [TextGen Ai WebUI Install! Run LLM Models in MINUTES! by Aitrepreneur](https://www.youtube.com/watch?v=lb_lC4XFedU) explains it in detail.

But not following automatic install on Windows has some advantages.
By using WSL2 on Windows 11 you can install Ubuntu inside your Windows 11 system and then installing TextGen WebUI. This supports faster Triton compiled GPTQ allowing to run act-order models. Xformers can also be used more easy than on Windows Native.

| Installation           | GPTQ Triton | GPTQ Cuda | xformers | act-order support |
| --- | --- | --- | --- | --- |
| Windows 11 WSL2 Ubuntu | yes         | yes       | yes      | yes               |
| Windows 11 Native      | no          | yes       | yes*     | no                |
| Ubuntu / Linux         | yes         | yes       | yes      | yes               |

Also, GPTQ Triton only supports 4 bit. If you want to use 3bit models, you need to use GPTQ Cuda

* manual installation

# Windows 11 WSL2 Ubuntu / Native Ubuntu

## Install Ubuntu WSL2 on Windows 11
1. Press the Windows key + X and click on "Windows PowerShell (Admin)" or "Windows Terminal (Admin)" to open PowerShell or Terminal with administrator privileges.
1. `wsl --install` You may be prompted to restart your computer. If so, save your work and restart.
1. Install Windows Terminal from Windows Store
1. Install Ubuntu on Windows Store
1. Choose the desired Ubuntu version (e.g., Ubuntu 20.04 LTS) and click "Get" or "Install" to download and install the Ubuntu app.
1. Once the installation is complete, click "Launch" or search for "Ubuntu" in the Start menu and open the app.
1. When you first launch the Ubuntu app, it will take a few minutes to set up. Be patient as it installs the necessary files and sets up your environment.
1. Once the setup is complete, you will be prompted to create a new UNIX username and password. Choose a username and password, and make sure to remember them, as you will need them for future administrative tasks within the Ubuntu environment.
1. If you prefer to use Windows Terminal from now on, close this console and start Windows Terminal then open a new Ubuntu console by clicking the drop down icon on top of Terminal and choose Ubuntu. Otherwise stay in the existing console window.

## Install Anaconda + Build Essentials
1. `sudo apt update`
2. `sudo apt upgrade`
2. `sudo apt install git`
2. `sudo apt install wget`
3. `mkdir downloads`
4. `cd downloads/`
5. `wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh`
6. `chmod +x ./Anaconda3-2022.05-Linux-x86_64.sh`
7. `./Anaconda3-2022.05-Linux-x86_64.sh` and follow the defaults
8. `sudo apt install build-essential`
9. `cd ..`

## Install text-generation-webui

1. `conda create -n textgen python=3.10.9`
1. `conda activate textgen`
1. `pip3 install torch torchvision torchaudio`
1. `mkdir github`
1. `cd github`
1. `git clone https://github.com/oobabooga/text-generation-webui`
1. `cd text-generation-webui`
1. `pip install -r requirements.txt`
1. `pip install chardet cchardet`

## Build and install GPTQ

If you want to try the triton branch, skip to [Newer GPTQ-Triton](#newer-gptq-triton)

### Older GPTQ-Cuda fork by pobabooga
- Works on Windows, Linux, WSL2.
- Supports 3 & 4 bit models
- Only supports no-act-order models
- Slower than triton
- Works best with `--groupsize 128 --wbits 4` and no-act-order models

1. `mkdir repositories`
1. `cd repositories`
1. `git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda` (or try the newer `https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda` build)
1. `cd GPTQ-for-LLaMa`
1. `python -m pip install -r requirements.txt`
1. `python setup_cuda.py install` if this gives an error about g++, try installing the correct g++ version: `conda install -y -k gxx_linux-64=11.2.0`
1. `cd ../..`

### Newer GPTQ-Triton
This [triton branch](https://github.com/qwopqwop200/GPTQ-for-LLaMa) or [this one](https://github.com/fpgaminer/GPTQ-triton):
- Works on Linux and WSL2
- Supports 4 bit quantized models
- Is faster than cuda
- Works best with the `--groupsize 128 --wbits 4` flags and act-order models

1. `mkdir repositories`
1. `cd repositories`
1. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
1. `git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa` (or try `https://github.com/fpgaminer/GPTQ-triton`)
1. `cd GPTQ-for-LLaMa`
1. `pip install -r requirements.txt`
1. `cd ../..`

### AutoGPTQ to install any (Newer Cuda, Newer Triton, older Cuda)
Alternatively you can try [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) to install cuda, older llama-cuda, or triton variants:

1. run one of these:
  - `pip install auto-gptq` to install cuda branch for newer models
  - `pip install auto-gptq[llama]` if your transformers is outdated or you are using older models that don't support it
  - `pip install auto-gptq[triton]` to install triton branch for triton compatible models
1. `cd ../..`

## LAN port forwarding from Ubuntu WSL

If you want to open the webui from within your home network, enable port forwarding on your windows machine, with this command in an administrator terminal:
`netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7860 connectaddress=localhost connectport=7860`

## Install bitsandbytes cuda
- Either always run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib` before running the sever.py below
- Or trying to install `pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113`

## Install xformers

Allows for faster, but non-deterministic inference. Optional:

- `pip install xformers`
- then use the `--xformers` flag later, when running the server.py below

You're done with the Ubuntu / WSL2 installation, you can skip to [Download models](#download-models) section.

# Windows 11 native

## Install Miniconda
1. Download and install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
1. Download and install [git for windows](https://git-scm.com/download/win)
2. Open `Anaconda Prompt (Miniconda 3)` from the Start Menu

## Install text-generation-webui
1. It should load in `C:\Users\yourusername>`
1. `mkdir github`
1. `cd github`
1. `conda create --name textgen python=3.10`
1. `conda activate textgen`
1. `conda install pip`
1. `conda install -y -k pytorch[version=2,build=py3.10_cuda11.7*] torchvision torchaudio pytorch-cuda=11.7 cuda-toolkit ninja git -c pytorch -c nvidia/label/cuda-11.7.0 -c nvidia`
1. `git clone https://github.com/oobabooga/text-generation-webui.git`
1. `python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`
1. `cd text-generation-webui`
1. `pip install -r requirements.txt --upgrade`
1. `mkdir repositories`
1. `cd repositories`
1. `git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda`
1. `python -m pip install -r requirements.txt`
1. `python setup_cuda.py install` might fail, continue with the next command if so
1. `pip install https://github.com/jllllll/GPTQ-for-LLaMa-Wheels/raw/main/quant_cuda-0.0.0-cp310-cp310-win_amd64.whl` skip this command, if the previous one didn't fail
1. `cd ..\..\..\` (go back to text-generation-webui)

# Download models

1. Still in your terminal, make sure you are in the /text-generation-webui/ folder and type `python download-model.py`
1. select other to download a custom model
1. paste the huggingface user/directory, for example: `TheBloke/wizardLM-7B-GGML` and let it download the model files

# Run
The base command to run. You have to add further flags, depending on the model and environment you want to run in:
1. if you are on WSL2 Ubuntu, run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib` always, before running the server.py
1. `python server.py --model-menu --chat`

- `--model-menu` to allow the change of models in the UI
- `--chat` loads the chat instead of the text completion UI
- `--wbits 4` loads a 4-bit quantized model
- `--groupsize 128` if the model specifies groupsize, add this parameter
- `--model_type llama` if the model name is unknown, specify it's base model. if you run llama derrived models like vicuna, alpaca, gpt4-x, codecapybara or wizardLM you have to define it as `llama`. If you load OPT or GPT-J models, define the flag accordingly
- `--xformers` if you have properly installed xformers and want faster but nondeterministic answer generation

# Troubleshoot

## cuda lib not found
If you get a `cuda lib not found` error, especially on Windows WSL2 Ubuntu, try executing `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib` before running the server.py above

## No GPU support on bitsandbytes
On Windows Native, try:
- `pip uninstall bitsandbytes`
- `pip install git+https://github.com/Keith-Hon/bitsandbytes-windows.git`
- [here](https://github.com/oobabooga/text-generation-webui/issues/1164) are some discussion, but some solutions are for Windows WSL2, some for Windows native

Or try these prebuilt wheel on windows:
- https://github.com/TimDettmers/bitsandbytes/files/11084955/bitsandbytes-0.37.2-py3-none-any.whl.zip
- https://github.com/acpopescu/bitsandbytes/releases/tag/v0.37.2-win.0
- And more help on windows support [here](https://github.com/TimDettmers/bitsandbytes/issues/30) and [here](https://github.com/oobabooga/text-generation-webui/issues/1164)

Still having problems, try to [manually](https://github.com/TimDettmers/bitsandbytes/issues/30#issuecomment-1455993902) copy the libraries

On Linux or Windows WSL2 Ubuntu, try:
- make sure you run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib` before running the server.py every time!
- alternatively, you can try `pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113` and see if it works without the above command


## Install xformers prebuilt Windows wheels 
- `pip install xformers==0.0.16rc425`

## Prebuilt GPTQ Windows Wheels (may be outdated)
- [GPTQ Wheels for Windows](https://github.com/jllllll/GPTQ-for-LLaMa-Wheels)

# Apple Silicon

Use [llama.cpp](https://github.com/ggerganov/llama.cpp), [HN discussion](https://news.ycombinator.com/item?id=35100086)

# Resources

## 3rd party models

See an up to date list of most models you can run locally: [awesome-ai open-models](https://github.com/underlines/awesome-marketing-datascience/blob/master/awesome-ai.md#open-models)


## Other tools

See the [awesome-ai LLM section](https://github.com/underlines/awesome-marketing-datascience/blob/master/awesome-ai.md#llm-guis) for more tools, GUIs etc.

## Other resources
- [LocalLLaMA on Reddit](https://reddit.com/r/localllama)
- [News about Llama](https://github.com/shm007g/LLaMA-Cult-and-More)
- [StackLLaMA: How to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)
- [Reddit LocalLLaMA model card](https://old.reddit.com/r/LocalLLaMA/wiki/models)
- [Reddit Oobabooga's Textgen subreddit](https://www.reddit.com/r/Oobabooga/)



