# Table of Contents

<details>
  <summary>Show Table of Contents</summary>
  
- [:speech_balloon: Large Language Models](#large-language-models)
  - [:open_file_folder: Open models](#open-models)
  - [:brain: Other SOTA Open Source Models](#other-sota-open-source-models)
  - [:floppy_disk: Data sets](#data-sets)
  - [:test_tube: Research](#research)
  - [:desktop_computer: LLM GUIs](#llm-guis)
    - [:rocket: OpenAI](#openai)
    - [:computer: Other GUIs](#other-guis)
  - [:package: LLM Wrappers](#llm-wrappers)
  - [:clapper: Showcases](#showcases)
  - [:wrench: Fine Tuning](#fine-tuning)
  - [:link: Other awesome lists](#other-awesome-lists)
- [:camera: Image Generation](#image-generation)
  - [:paintbrush: Models](#models)
  - [:computer_mouse: Wrappers & GUIs](#wrappers--guis)
  - [:wrench: Fine Tuning](#fine-tuning-1)
- [:stopwatch: Benchmarking](#benchmarking)
- [:movie_camera: Video](#video)
  - [:memo: Text to video generation](#text-to-video-generation)
  - [:film_strip: Frame Interpolation (Temporal Interpolation)](#frame-interpolation-temporal-interpolation)
  - [:mag_right: Super Resolution (Spacial Interpolation)](#super-resolution-spacial-interpolation)
  - [:clapper: Spacio Temporal Interpolation](#spacio-temporal-interpolation)
- [:musical_note: Audio](#audio)
  - [:clamp: Compression](#compression)
  - [:headphones: Multiple Tasks](#multiple-tasks)
  - [:microphone: Speech Recognition](#speech-recognition)
  - [:notes: TextToSpeech](#texttospeech)
  - [:loud_sound: Voice Conversion](#voice-conversion)
- [:computer: AI DevOps](#ai-devops)
- [:chart_with_upwards_trend: Optimization](#optimization)
  - [:bar_chart: Inference](#inference)
  - [:running: Training](#training)
  - [:wrench: Other Optimization](#other-optimization)

</details>



# Large Language Models

## Open models

The model list moved [here](llm-model-list.md)


## Other SOTA Open Source models
- [Cerebras GPT-13b](https://huggingface.co/cerebras) ([release notes](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/))
- [LAION OpenFlamingo | Multi Modal Model and training architecture](https://github.com/mlfoundations/open_flamingo)
- [TheBloke/galpaca-30b-gptq-4bit-128g](https://huggingface.co/TheBloke/galpaca-30B-GPTQ-4bit-128g), GALACTICA 30B fine tuned with Alpaca 
- [GeorgiaTechResearchInstitute/galpaca-6.7b](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-6.7b) GALACTICA 6.7B fine tuned with Alpaca
- [GeoV/GeoV-9b](https://huggingface.co/GeoV/GeoV-9b) - 9B parameter, in-progress training to 300B tokens (33:1)
- [RWKV: Parallelizable RNN with Transformer-level LLM Performance](https://github.com/BlinkDL/RWKV-LM)
- [CodeGeeX 13B | Multi Language Code Generation Model](https://huggingface.co/spaces/THUDM/CodeGeeX)
- [BigCode | Open Scientific collaboration to train a coding LLM](https://huggingface.co/bigcode)
- [MOSS by Fudan University](https://github.com/OpenLMLab/MOSS)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) Multimodal finetuned model for visual/language tasks
- [Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT) multi-modal visual/language chatbot, using llama with custom LoRA weights and openflamingo-9B.
- [Visual-med-alpaca](https://github.com/cambridgeltl/visual-med-alpaca) fine-tuning llama-7b on self instruct for the biomedical domain. Models locked behind a request form.
- [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) focused on Code Completion. The model has been trained on a subset of the [Stack Dedup v1.2](https://arxiv.org/abs/2211.15533) dataset.
- [VPGTrans](https://vpgtrans.github.io/) Transfer Visual Prompt Generator across LLMs and the VL-Vicuna model is a novel VL-LLM. [Paper](https://arxiv.org/abs/2305.01278), [code](https://github.com/VPGTrans/VPGTrans)


## Data sets
- [Alpaca-lora instruction finetuned using Low Rank Adaption](https://github.com/tloen/alpaca-lora)
- [codealpaca Instruction training data set for code generation](https://github.com/sahil280114/codealpaca)
- [LAION AI / Open-Assistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) (https://github.com/LAION-AI/Open-Assistant / https://projects.laion.ai/Open-Assistant/ / https://open-assistant.io)
- [pre-cleaned, English only, "unfiltered," and 2048 token split version of the ShareGPT dataset ready for finetuning](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [Vicuna ShareGPT pre-cleaned 90k conversation dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset)
- [Vicuna ShareGPT unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [GPTeacher](https://github.com/teknium1/GPTeacher)
- [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [codealpaca 20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [gpt3all pruned](https://huggingface.co/datasets/Nebulous/gpt4all_pruned)
- [gpt4all_prompt_generations_with_p3](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations_with_p3)
- [gpt4all_prompt_generations](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations)
- [alpaca-plus-gpt4all-without-p3](https://huggingface.co/datasets/magicgh/alpaca-plus-gpt4all-without-p3)
- [Alpaca dataset from Stanford, cleaned and curated](https://github.com/gururise/AlpacaDataCleaned) 
- [Alpaca Chain of Thought fine tuning dataset for EN and CN](https://github.com/PhoebusSi/Alpaca-CoT)
- [PRESTO | Multilingual dataset for parsing realistic task-oriented dialogues by Google & University of Rochester, California, Santa Barbara, Columbia](https://ai.googleblog.com/2023/03/presto-multilingual-dataset-for-parsing.html) [paper](https://arxiv.org/pdf/2303.08954.pdf)
- [RedPajama](https://www.together.xyz/blog/redpajama) Dataset and model similar to LLaMA but truly open source and ready for commercial use. [hf](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- [BigCode The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [open-instruct-v1](https://huggingface.co/datasets/hakurei/open-instruct-v1)
- [list of instruction datasets by yadongC/awesome-instruction-dataset](https://github.com/yaodongC/awesome-instruction-dataset)
- [The Embedding Archives, Millions of Wikipedia Article Embeddings in multiple languages](https://txt.cohere.com/embedding-archives-wikipedia/)
- [Rereplit-finetuned-v1-3b & replit-code-v1-3b](https://twitter.com/Replit/status/1651344186715803648) outperforming all coding OSS models, gets released soon
- [alpaca_evol_instruct_70k](https://huggingface.co/datasets/victor123/evol_instruct_70k) an instruction-following dataset created using [Evol-Instruct](https://github.com/nlpxucan/evol-instruct), used to fine-tune [WizardLM](https://github.com/nlpxucan/WizardLM)
- [gpt4tools_71k.json](https://github.com/StevenGrove/GPT4Tools#Dataset) from GPT4Tools paper, having 71k instruction-following examples for sound/visual/text instructions
- [WizardVicuna 70k dataset](https://huggingface.co/datasets/junelee/wizard_vicuna_70k) used to fine tune [WizardVicuna](https://github.com/melodysdreamj/WizardVicunaLM)

## Research
- [LLM Model Cards](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs)
- [GPTs are GPTs: An early look at the labor market impact potential of LLMs](https://arxiv.org/abs/2303.10130)
- [ViperGPT Visual Inference via Python Execution for reasoning](https://viper.cs.columbia.edu/)
- [Emergent Abilities of LLMs ](https://openreview.net/forum?id=yzkSU5zdwD), [blog post](https://www.jasonwei.net/blog/emergence)
- [visualchatgpt | Microsoft research proposes a multi-modal architecture to give chatgpt the ability to interpret and generate images based on open source foundation models](https://github.com/microsoft/visual-chatgpt)
- [facts checker reinforcement](https://arxiv.org/abs/2302.12813)
- [LLaVA: Large Language and Vision Assistant, combining LLaMA with a visual model. Delta-weights released](https://llava-vl.github.io/)
- [Mass Editing Memory in a Transformer](https://memit.baulab.info/)
- [MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://minigpt-4.github.io/)
- [WizardLM | Fine tuned LLaMA 7B with evolving instructions, outperforming chatGPT and Vicuna 13B on complex test instructions](https://arxiv.org/abs/2304.12244) ([code](https://github.com/nlpxucan/WizardLM), [delta weights](https://huggingface.co/victor123/WizardLM))
- [Scaling Transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062)
- [AudioGPT | Understanding and Generating Speech, Music, Sound, and Talking Head](https://arxiv.org/abs/2304.12995) ([github](https://github.com/AIGC-Audio/AudioGPT), [hf space](https://huggingface.co/spaces/AIGC-Audio/AudioGPT))
- [Chameleon-llm](https://github.com/lupantech/chameleon-llm), a [paper](https://arxiv.org/abs/2304.09842) about Plug-and-Play Compositional Reasoning with GPT-4
- [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) share data generated by GPT-4 for building an instruction-following LLMs with supervised learning and reinforcement learning. [paper](https://arxiv.org/abs/2304.03277)
- [GPT4Tools](https://gpt4tools.github.io/) Teaching LLM to Use Tools via Self-instruct. [code](https://github.com/StevenGrove/GPT4Tools)
- [CAMEL](https://github.com/lightaime/camel): Communicative Agents for "Mind" Exploration of Large Scale Language Model Society. [preprint paper](https://ghli.org/camel.pdf), [website](https://www.camel-ai.org/)
- [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944)
- [SparseGPT](https://arxiv.org/abs/2301.00774): Massive Language Models Can Be Accurately Pruned in One-Shot
- [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199)
- [Dromedary](https://arxiv.org/abs/2305.03047): Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision, [code](https://github.com/IBM/Dromedary), [weights](https://huggingface.co/zhiqings/dromedary-65b-lora-delta-v0)
- [Unlimiformer](https://arxiv.org/abs/2305.01625): transformer-based model that can process unlimited length input by offloading attention computation to a k-nearest-neighbor index, extending the capabilities of existing models like BART and Longformer without additional weights or code modifications. [code](https://github.com/abertsch72/unlimiformer)
- [Salesforce LAVIS](https://github.com/salesforce/lavis) provides a comprehensive Python library for language-vision intelligence research, including state-of-the-art models like BLIP-2 for vision-language pretraining and Img2LLM-VQA for visual question answering, alongside a unified interface
- [FLARE](https://arxiv.org/abs/2305.06983v1) an active retrieval augmented generation technique that iteratively predicts, retrieves, and refines content, improving the accuracy and efficiency of long-form text generation in language models




## LLM GUIs

### OpenAI

-   [chatgptui/desktop](https://github.com/chatgptui/desktop)
  -   [chatbox](https://github.com/Bin-Huang/chatbox) is a Windows, Mac & Linux native ChatGPT Client
  -   [BingGPT](https://github.com/dice2o/BingGPT) Desktop application of new Bing's AI-powered chat
-   [TypingMind](https://www.typingmind.com/)
-   [Chatwithme.chat](https://www.chatwithme.chat/)
-   [datafilik/GPT-Voice-Assistant](https://github.com/datafilik/GPT-Voice-Assistant)
-   [Abdallah-Ragab/VoiceGPT](https://github.com/Abdallah-Ragab/VoiceGPT)
-   [LlmKira/Openaibot](https://github.com/LlmKira/Openaibot)
-   [chathub-dev/chathub](https://github.com/chathub-dev/chathub)
-   [enricoros/nextjs-chatgpt-app](https://github.com/enricoros/nextjs-chatgpt-app)
-   [no8081/chatgpt-demo](https://github.com/ddiu8081/chatgpt-demo)
-   [Auto GPT](https://github.com/Torantulino/Auto-GPT)
    -   [AgentGPT](https://github.com/reworkd/AgentGPT) Deploy autonomous AI agents, using vectorDB memory, web browsing via LangChain, website interaction and more 
    -   [microGPT ](https://github.com/muellerberndt/micro-gpt) Autonomous GPT-3.5/4 agent, can analyze stocks, create art, order pizza, and perform network security tests
    -   [Auto GPT Plugins](https://github.com/Significant-Gravitas/Auto-GPT-Plugins)
    -   [AutoGPT-Next-Web](https://github.com/Dogtiti/AutoGPT-Next-Web) An AgentGPT fork as a Web GUI
    -   [AutoGPT Web](https://github.com/jina-ai/auto-gpt-web)
    -   [AutoGPT.js](https://github.com/zabirauf/AutoGPT.js)
    -   [LoopGPT](https://github.com/farizrahman4u/loopgpt) a re-implementation of AutoGPT as a proper python package, modular and extensible
    -   [Camel-AutoGPT](https://github.com/SamurAIGPT/Camel-AutoGPT) Communicaton between Agents like BabyAGI and AutoGPT
    -   [BabyAGIChatGPT](https://github.com/Doriandarko/BabyAGIChatGPT) is a fork of BabyAGI to work with OpenAI's GPT, pinecone and google search
    -   [GPT Assistant](https://github.com/BuilderIO/gpt-assistant) An autonomous agent that can access and control a chrome browser via Puppeteer 
    -   [gptchat](https://github.com/ian-kent/gptchat) a client which uses GPT-4, adding long term memory, can write its own plugins and can fulfill tasks
    -   [Chrome-GPT](https://github.com/richardyc/Chrome-GPT) experimental AutoGPT agent employing Langchain and Selenium to interact with and manipulate a Chrome browser session, enabling capabilities like Google search, webpage description, element interaction, and form input
-   [cheetah | Speech to text for remote coding interviews, giving you hints from GTP3/4](https://github.com/leetcode-mafia/cheetah)
-   [sqlchat | Use OpenAI GPT3/4 to chat with your database](https://github.com/sqlchat/sqlchat)
-   [chat-with-github-repo](https://github.com/peterw/Chat-with-Github-Repo) which uses streamlit, gpt3.5-turbo and deep lake to answer questions about a git repo
-   [BarkingGPT | Audio2Audio by using Whisper+chatGPT+Bark](https://github.com/BudEcosystem/BarkingGPT)
  -  [gpt_chatbot](https://github.com/1nnovat1on/gpt_chatbot) Windows / elevenlabs TTS + pinecone long term memory
  -  [gpt-voice-conversation-chatbot](https://github.com/Adri6336/gpt-voice-conversation-chatbot) using GPT3.5/4 API, elevenlab voices, google tts, session long term memory
  -  [JARVIS-ChatGPT](https://github.com/gia-guar/JARVIS-ChatGPT) conversational assistant that uses OpenAI Whisper, OpenAI ChatGPT, and IBM Watson to provide quasi-real-time tips and opinions.
-   [ALFRED | LangChain Voice Assistant, powered by GPT-3.5-turbo, whisper, Bark, pyttsx3 and more](https://github.com/masrad/ALFRED)
-   [IPython-gpt](https://github.com/santiagobasulto/ipython-gpt) use chatGPT directly inside jupyter notebooks
-   [Glarity](https://github.com/sparticleinc/chatgpt-google-summary-extension) open-source chrome extension to write summaries for various websites including custom ones and YouTube videos. Extensible
  -   [superpower-chatgpt](https://github.com/saeedezzati/superpower-chatgpt) chrome  extension / firefox addon to add missing features like Folders, Search, and Community Prompts to ChatGPT



### Other GUIs

-   [Text Generation Webui](https://github.com/oobabooga/text-generation-webui) An all purpose UI to run LLMs of all sorts with optimizations
    -   [running LLaMA on less than 10GB vram](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134)
    -   [running LLaMA-7b on a 3080](https://github.com/TimDettmers/bitsandbytes/issues/30#issuecomment-1455993902)
    -   [running LLaMA-13b on 6GB VRAM](https://gist.github.com/rain-1/8cc12b4b334052a21af8029aa9c4fafc), [HN Thread](https://news.ycombinator.com/item?id=35937505)
-   [Alpaca-LoRa-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)
-   [llama.cpp](https://github.com/ggerganov/llama.cpp)
    -   [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
    -   [koboldcpp](https://github.com/LostRuins/koboldcpp)
    -   [Serge](https://github.com/nsarrazin/serge) chat interface based on llama.cpp for running Alpaca models. Entirely self-hosted, no API keys needed
    -   [llama MPS](https://github.com/jankais3r/LLaMA_MPS) inference on Apple Silicon GPU using much lower power but is slightly slower than llama.cpp which uses CPU
-   [Dalai](https://github.com/cocktailpeanut/dalai)
-   [ChatLLaMA | LLaMA-based ChatGPT for single GPUs](https://github.com/juncongmoo/chatllama)
    -   [ChatLLaMA | another implementation](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)
-   [Chatbot web app + HTTP and Websocket endpoints for BLOOM-176B inference with the Petals client](https://github.com/borzunov/chat.petals.ml)
    -   [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp) Inference of HuggingFace's BLOOM-like models in pure C/C++
-   [Vicuna FastChat](https://github.com/lm-sys/FastChat)
-   [Lit-llama](https://github.com/Lightning-AI/lit-llama)
-   [gpt4all](https://github.com/nomic-ai/gpt4all) terminal and gui version to run local gpt-j models, [compiled binaries for win/osx/linux](https://gpt4all.io/index.html)
    -   [gpt4all.zig](https://github.com/renerocksai/gpt4all.zig) terminal version of GPT4All
    -   [gpt4all-chat](https://github.com/nomic-ai/gpt4all-chat) Cross platform desktop GUI for GPT4All  models (gpt-j)
-   [openplayground Try out almost any LLM in a gui](https://github.com/nat/openplayground)
-   [Alpaca-Turbo | Web UI to run alpaca model locally on Win/Mac/Linux](https://github.com/ViperX7/Alpaca-Turbo)
-   [FreedomGPT | Web app that executes the FreedomGPT LLM locally](https://github.com/ohmplatform/FreedomGPT)
-   [Auto Vicuna Butler](https://github.com/NiaSchim/auto-vicuna-butler) Baby-AGI fork / AutoGPT alternative to run with local LLMs
    -   [BabyAGI](https://github.com/yoheinakajima/babyagi) AI-Powered Task Management for OpenAI + Pinecone or Llama.cpp
    -   [Agent-LLM](https://github.com/Josh-XT/Agent-LLM) Webapp to control an agent-based Auto-GPT alternative, supporting GPT4, Kobold, llama.cpp, FastChat, Bard, Oobabooga textgen
    -   [auto-llama-cpp](https://github.com/rhohndorf/Auto-Llama-cpp) fork of Auto-GPT with added support for locally running llama models through llama.cpp
    -   [AgentOoba](https://github.com/flurb18/AgentOoba) autonomous AI agent extension for Oobabooga's web ui
-   [huggingGPT / JARVIS](https://github.com/microsoft/JARVIS) Connects LLMs with huggingface specialized models
-   [OpenAGI](https://github.com/agiresearch/openagi) AGI research platform, solves multi step tasks with RLTF and supports complex model chains
-   [bark TTS for oobabooga/text-generation-webui](https://github.com/wsippel/bark_tts) make your local LLM talk
    -   [bark TTS for oobabooga/text-generation-webui](https://github.com/minemo/text-generation-webui-barktts) another implementation
-   [HuggingChat](https://huggingface.co/chat) open source chat interface for transformer based LLMs by Huggingface
-   [mlc-llm](https://github.com/mlc-ai/mlc-llm), run any LLM on any hardware (iPhones, Android, Win, Linux, Mac, WebGPU, Metal. NVidia, AMD)
-   [faraday.dev](https://faraday.dev/) Run open-source LLMs on your Win/Mac. Completely offline. Zero configuration.
-   [PrivateGPT](https://github.com/imartinez/privateGPT) a standalone question-answering system using LangChain, GPT4All, LlamaCpp and embeddings models to enable offline querying of documents

## LLM Wrappers

-   [acheong08/ChatGPT Python](https://github.com/acheong08/ChatGPT)
-   [mpoon/gpt-repository-loader](https://github.com/mpoon/gpt-repository-loader)
-   [LangChain | framework for developing LLM applications](https://github.com/hwchase17/langchain) ([example](https://www.youtube.com/watch?v=iRJ4uab_NIg&t=588s), [paolorechia/learn-langchain with vicuna and GPQT 4 bit support](https://github.com/paolorechia/learn-langchain))
-   [LangFlow | GUI for Langchain](https://github.com/logspace-ai/langflow)
-   [pyllama | hacked version of LLaMA based on Meta's implementation, optimized for Single GPUs](https://github.com/juncongmoo/pyllama)
-   [Toolformer implementation | Allows LLMs to use Tools](https://github.com/lucidrains/toolformer-pytorch)
-   [FastLLaMA Python wrapper for llama.cpp](https://github.com/PotatoSpudowski/fastLLaMa)
-   [LlamaIndex](https://github.com/jerryjliu/llama_index) provides a central interface to connect your LLM's with external data
    -   [Llama-lab](https://github.com/run-llama/llama-lab) home of llama_agi and auto_llama using LlamaIndex
-   [Paper QA](https://github.com/whitead/paper-qa) LLM Chain for answering questions from documents with citations, using OpenAI Embeddings or local llama.cpp, langchain and FAISS Vector DB
-   [supercharger | Write Software + unit tests for you, based on Baize-30B 8bit, using model parallelism](https://github.com/catid/supercharger)
-   [WebGPT Inference in pure javascript](https://github.com/0hq/WebGPT)
-   [WasmGPT ChatGPT-like chatbot in browser using ggml and emscripten](https://github.com/lxe/ggml/tree/wasm-demo)
-   [FauxPilot | open source Copilot alternative using Triton Inference Server](https://github.com/fauxpilot/fauxpilot)
    -   [Turbopilot | open source LLM code completion engine and Copilot alternative](https://github.com/ravenscroftj/turbopilot)
    -   [Tabby | Self hosted Github Copilot alternative](https://github.com/TabbyML/tabby)
-   [Sidekick | Information retrieval for LLMs](https://github.com/ai-sidekick/sidekick)
-   [gpt4free | Use reverse engineered GPT3.5/4 APIs of other website's APIs](https://github.com/xtekky/gpt4free)
-   [AutoGPTQ | easy-to-use model GPTQ quantization package with user-friendly CLI](https://github.com/PanQiWei/AutoGPTQ)
-   [RWKV.cpp](https://github.com/saharNooby/rwkv.cpp) CPU only port of BlinkDL/RWKV-LM to ggerganov/ggml. Supports FP32, FP16 and quantized INT4.
    -   [RWKV Cuda](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda) a torchless, c++ rwkv implementation with 8bit quantization written in cuda
-   [gpt-llama.cpp](https://github.com/keldenl/gpt-llama.cpp) Replace OpenAi's GPT APIs with llama.cpp's supported models locally
-   [llama-node](https://github.com/Atome-FE/llama-node) JS client library for llama (or llama based) LLMs built on top of llama-rs and llama.cpp.
-   [megabots](https://github.com/momegas/megabots) to create LLM bots by providing Q&A, document retrieval, vector DBs, FastAPI, Gradio UI, GPTCache, guardrails, whisper, supports OpenAI API (local LLMs planned)
-   [GPTCache](https://github.com/zilliztech/GPTCache), serve cached results based on embeddings in a vector DB, before querying the OpenAI API.
-   [kitt](https://github.com/livekit-examples/kitt) TTS + GPT4 + STT to create a conference call audio bot
-   [Jsonformer](https://github.com/1rgs/jsonformer): Generate Structured JSON from Language Models by handling JSON synthax, and letting LLM just output the values
-   [Marvin](https://github.com/prefecthq/marvin) simplifies AI integration in software development with easy creation of AI functions and bots managed through a conversational interface
-   [chatgpt.js](https://github.com/chatgptjs/chatgpt.js) client-side JavaScript library for ChatGPT
-   [ChatGPT-Bridge](https://github.com/improveTheWorld/ChatGPT-Bridge) use chatGPT plus' GPT-4 as a local API


## Showcases
- [Opinionate.io AI Debating AI](https://opinionate.io/)
- [phind.com](phind.com) Developer Search Engine
- [Voice Q&A Assistant](https://github.com/hackingthemarkets/qa-assistant-eleven-labs-voice-cloning) using ChatGPT API, Embeddings, Gradio, Eleven Labs and Whisper
- [chatpdf](https://www.chatpdf.com/), Q&A for PDFs

## Fine Tuning
- [simple llama finetuner](https://github.com/lxe/simple-llama-finetuner)
- [LLaMA-LoRA Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner)
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
- [StackLLaMA Fine-Tuning Guide by huggingface](https://huggingface.co/blog/stackllama)
- [xTuring | LLM finetuning pipeline supporting LoRa & 4bit](https://github.com/stochasticai/xturing)
- [Microsoft DeepSpeed Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md)
- [How to train your LLMs](https://blog.replit.com/llm-training)
- [H2O LLM Studio | Framework and no-code GUI for fine tuning SOTA LLMs](https://github.com/h2oai/h2o-llmstudio)
- [Implementation of LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter), to fine tune instructions within hours

## Other awesome resources
- [LLM Worksheet](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=741531996) by [randomfoo2](https://www.reddit.com/r/LocalAI/comments/12smsy9/list_of_public_foundational_models_fine_tunes/)
- [The full story of LLMs](https://www.assemblyai.com/blog/the-full-story-of-large-language-models-and-rlhf/)
- [Brief history of llama models](https://agi-sphere.com/llama-models/)
- [A timeline of transformer models](https://ai.v-gar.de/ml/transformer/timeline/)
- [Every front-end GUI client for ChatGPT API](https://github.com/billmei/every-chatgpt-gui)
- [LLM Logic Tests](https://docs.google.com/spreadsheets/d/1NgHDxbVWJFolq8bLvLkuPWKC7i_R6I6W/edit#gid=719051075) by [YearZero on reddit/localllama](https://www.reddit.com/r/LocalLLaMA/comments/13636h5/updated_riddlecleverness_comparison_of_popular/)
- [LLMSurvey](https://github.com/rucaibox/llmsurvey) a collection of papers and resources including an LLM timeline
- [rentry.org/lmg_models](https://rentry.org/lmg_models) a list of llama derrivates and models
- [Timeline of AI and language models](https://lifearchitect.ai/timeline/) and [Model Comparison Sheet](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs/edit#gid=1158069878) by Dr. Alan D. Thompson
- [Brex's Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering) an evolving manual providing historical context, strategies, guidelines, and safety recommendations for building programmatic systems on OpenAI's GPT-4
- [LLMs Practical Guide](https://github.com/Mooler0410/LLMsPracticalGuide) actively curated collection of a timeline and guides for LLMs, providing a historical context and restrictions based on [this](https://arxiv.org/abs/2304.13712) paper and community contributions
- [LLMSurvey](https://github.com/RUCAIBox/LLMSurvey) based on [this](https://arxiv.org/abs/2303.18223) paper, builds a collection of further papers and resources related to LLMs including a timeline
- [LLaMAindex](https://medium.com/llamaindex-blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec) using Document Summary Index for better QA performance compared to vectorDBs


# Audio
## Compression
- https://github.com/facebookresearch/encodec

## Multiple Tasks
- [Speechbrain](https://github.com/speechbrain/speechbrain) A PyTorch-based Speech Toolkit for TTS, STT, etc
- [Nvidia NeMo](https://github.com/NVIDIA/NeMo) TTS, LLM, Audio Synthesis framework
- [speech-rest-api](https://github.com/askrella/speech-rest-api) for Speech-To-Text and Text-To-Speech with Whisper and Speechbrain
- [LangHelper](https://github.com/NsLearning/LangHelper) language learning through Text-to-speech + chatGPT + speech-to-text to practise speaking assessments, memorizing words and listening tests
- [Silero-models](https://github.com/snakers4/silero-models) pre-trained speech-to-text, text-to-speech and text-enhancement for ONNX, PyTorch, TensorFlow, SSML
- [AI-Waifu-Vtuber](https://github.com/ardha27/AI-Waifu-Vtuber) AI Waifu Vtuber & is a virtual streamer. Supports multiple languages and uses VoiceVox, DeepL, Whisper, Seliro TTS, and VtubeStudio, and now also supports Twitch streaming.


## Speech Recognition
- https://github.com/openai/whisper
  - [Whisper JAX implementation](https://github.com/sanchit-gandhi/whisper-jax) runs around 70x faster on CPU, GPU and TPU
  - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) C/C++ port for Intel and ARM based Mac OS, ANdroid, iOS, Linux, WebAssembly, Windows, Raspberry Pi
- [ermine-ai | Whisper in the browser using transformers.js](https://github.com/vishnumenon/ermine-ai)
- [wav2vec2 dimensional emotion model](https://github.com/audeering/w2v2-how-to)
- [MeetingSummarizer](https://github.com/rajpdus/MeetingSummarizer) using Whisper and GPT3.dd

## TextToSpeech
-   [tts-generation-webui](https://github.com/rsxdalv/tts-generation-webui) for all things TTS, currently supports Bark
-   [Bark](https://github.com/suno-ai/bark) transformer-based text-to-audio model by Suno. Can generate highly realistic, multilingual speech and other audio like music, background noise and simple effects
    -   [Bark-Voice-Clones](https://github.com/nikaskeba/Bark-Voice-Clones)
    -   [Bark WebUI colab notebooks](https://github.com/camenduru/bark-colab)
    -   [bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone)
    -   [Bark Infinity for longer audio](https://github.com/JonathanFly/bark)
    -   [Bark WebUI](https://github.com/makawy7/bark-webui))
-   [Coqui TTS | deep learning toolkit for Text-to-Speech](https://github.com/coqui-ai/TTS)
    -   [Tutorial](https://www.youtube.com/watch?v=dfmlyXHQOwE) for Coqui VITS and Whisper to automate voice cloning and [Colab notebook](https://colab.research.google.com/drive/1Swo0GH_PjjAMqYYV6He9uFaq5TQsJ7ZH?usp=sharing#scrollTo=nSrZbKCXxalg)
-   [StyleTTS implementation](https://github.com/yl4579/StyleTTS)
  -   [StyleTTS-VC](https://github.com/yl4579/StyleTTS-VC) One-Shot Voice Conversion by Knowledge Transfer from Style-Based TTS Models
-   [Vall-E and Vall-E X](https://valle-demo.github.io/), [paper](https://arxiv.org/abs/2301.02111), [code](https://github.com/enhuiz/vall-e). Zero Shot TTS preserving emotion, expression, similarity and allows language transfer
  -  [Vall-e PyTorch Implementation](https://github.com/enhuiz/vall-e) of Vall-E based on EnCodec tokenizer
  -  [Vall-E PyTorch implementation](https://github.com/lifeiteng/vall-e)
-   [NaturalSpeech implmenetation](https://github.com/heatz123/naturalspeech)
-   [IMS Toucan, TTS Toolkit from University of Stuttgart](https://github.com/digitalphonetics/ims-toucan)
-   [YourTTS | Zero Shot Multi Speaker TTS and Voice Conversion for everyone](https://github.com/Edresson/YourTTS)
-   [PaddleSpeech | Easy to use Speech Toolkit with Self Supervised learning, SOTA Streaming with punctuation, TTS, Translation etc](https://github.com/PaddlePaddle/PaddleSpeech)
-   [Tortoise TTS | Open source multi voice TTS system](https://github.com/neonbjb/tortoise-tts)
    -   [finetune guide using DLAS DL-Art-School](https://www.youtube.com/watch?v=lnIq4SFFXWs)
    -   [DL-Art-School](https://github.com/152334H/DL-Art-School) fine tuning tortoise with DLAS GUI
    -   [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) fast Tortoise TTS inference up to 5x. [Video tutorial](https://www.youtube.com/watch?v=8i4T5v1Fl_M)
    -   [Tortoise mrq fork for voice cloning](https://git.ecker.tech/mrq/ai-voice-cloning)
-   [piper](https://github.com/rhasspy/piper) A fast, local neural text to speech system that sounds great and is optimized for the Raspberry Pi 4. Using VITS and onnxruntime
-   [PITS](https://github.com/anonymous-pits/pits) PyTorch implementation of Variational Pitch Inference for End-to-end Pitch-controllable TTS. [hf demo](https://huggingface.co/spaces/anonymous-pits/pits), [samples](https://anonymous-pits.github.io/pits/)
-   [VoiceCloning](https://github.com/MartinMashalov/VoiceCloning) Implementing the [YourTTS paper](https://arxiv.org/abs/2112.02418) for Zero-Shot multi-speaker Attention-Based TTS using VITS approaches
  -   [VITS-Umamusume-voice-synthesizer](https://huggingface.co/spaces/1raliopunche/VITS-Umamusume-voice-synthesizer) (Multilingual Anime TTS) Including Japanese TTS, Chinese and English TTS, speakers are all anime characters.
-   [Parallel WaveGAN implementation in PyTorch](https://github.com/kan-bayashi/ParallelWaveGAN) for high quality text to speech synthesis [paper](https://github.com/kan-bayashi/ParallelWaveGAN)


## Voice Conversion

-   [voicepaw/so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) SoftVC VITS Singing Voice Conversion Fork with realtime support and greatly improved interface. Based on so-vits-svc 4.0 (v1)
    -   [Video tutorial by Nerdy Rodent](https://www.youtube.com/watch?v=tZn0lcGO5OQ)
    -   [nateraw/so-vits-svc-fork gradio app](https://github.com/nateraw/voice-cloning) for inference of so-vits-svc-fork voice models + ([training in colab](https://colab.research.google.com/github/nateraw/voice-cloning/blob/main/training_so_vits_svc_fork.ipynb) with yt downloader and audio splitter, [hf space demo](https://hf.co/spaces/nateraw/voice-cloning))
    -   [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
    -   [LoRa svc](https://github.com/PlayVoice/lora-svc)
    -   [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) simple and easy-to-use voice transformation (voice changer) web GUI based on VITS
        -   [rvc-webui](https://github.com/ddPn08/rvc-webui) Win/Mac/Linux installer and Guide for RVC-Project
    -   [w-okada/voice-changer](https://github.com/w-okada/voice-changer) supports various models: MMVC, so-vits-svc, RVC, DDSP-SVC, can work over LAN to offload processing
    -   [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) Real-time singing voice conversion based on DDSP, training and inference uses lower requirements than diff-svc and so-vits-svc
    -   [Leader board of SOTA models](https://github.com/Anjok07/ultimatevocalremovergui/issues/344) for stem separation using model ensembles in UVR
    -   [VITS GUI to load VITS text to speech models](https://github.com/CjangCjengh/MoeGoe_GUI)
    -   [Vits-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning) pipeline of VITS finetuning for fast speaker adaptation TTS, and many-to-many voice conversion
-   [w-okada/voice-changer | real time voice conversion using various models like MMVC, so-vits-svc, RVC, DDSP-SVC](https://github.com/w-okada/voice-changer/blob/master/README_en.md)
-   [Diff-svc](https://github.com/prophesier/diff-svc) Singing Voice Conversion via Diffusion model
  -   [FastDiff implementation| Fast Conditional Diffusion Model for High-Quality Speech Synthesis](https://github.com/Rongjiehuang/FastDiff)
  -   [Fish Diffusion](https://github.com/fishaudio/fish-diffusion) easy to understand TTS / SVS / SVC framework, can convert Diff models
-   [Auto-synced-translated-dubs](https://github.com/ThioJoe/Auto-Synced-Translated-Dubs) Youtube audio translation and dubbing pipeline using Whisper speech-to-text, Google/DeepL Translate, Azure/Google TTS
-   [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) abandoned project
    -   [Real-Time-Voice-Cloning v2](https://github.com/liuhaozhe6788/voice-cloning-collab) active fork of the original
-  [Raven with voice cloning 2.0](https://huggingface.co/spaces/Kevin676/Raven-with-Voice-Cloning-2.0/tree/main) by Kevin676
-  [CoMoSpeech](https://paperswithcode.com/paper/comospeech-one-step-speech-and-singing-voice) paper, consistency model distilled from a diffusion-based teacher model, enabling high-quality one-step speech and singing voice synthesis


# AI DevOps
- https://www.steamship.com/
- [SynapseML](https://github.com/microsoft/SynapseML) (previously known as MMLSpark),an open-source library that simplifies the creation of massively scalable machine learning (ML) pipelines

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
- https://petals.ml/
- https://github.com/facebookincubator/AITemplate

## Training
- https://github.com/learning-at-home/hivemind

## Other Optimization
- https://github.com/HazyResearch/flash-attention
- https://github.com/stochasticai/x-stable-diffusion
- [Long Convs and Hyena implementations](https://github.com/hazyresearch/safari) 


# Benchmarking
- https://videoprocessing.ai/benchmarks/
- https://paperswithcode.com/
- [Pythia | interpretability analysis for autoregressive transformers during training](https://github.com/EleutherAI/pythia)
- [LMSys Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard), [blogpost](https://lmsys.org/blog/2023-05-03-arena/) is an anonymous benchmark platform for LLMs that features randomized battles in a crowdsourced manner
- [Current best choices](https://old.reddit.com/r/LocalLLaMA/wiki/models#wiki_current_best_choices) on LocalLLaMA reddit


# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=underlines/awesome-marketing-datascience&type=Date)](https://star-history.com/#underlines/awesome-marketing-datascience&Date)
