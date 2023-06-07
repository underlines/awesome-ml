# Tools
## Native GUIs
### openAI
-   [chatgptui/desktop](https://github.com/chatgptui/desktop)
-   [chatbox](https://github.com/Bin-Huang/chatbox) is a Windows, Mac & Linux native ChatGPT Client
-   [BingGPT](https://github.com/dice2o/BingGPT) Desktop application of new Bing's AI-powered chat
-   [cheetah](https://github.com/leetcode-mafia/cheetah) Speech to text for remote coding interviews, giving you hints from GTP3/4
### Local LLMs
cpp / ggml:
-   [llama.cpp](https://github.com/ggerganov/llama.cpp) runs ggml models up to 4-bit quantized on mac, linux and windows natively. Supports the new ggmlv3 format and runs on CPU and GPU. Allows for mixed use of CPU/GPU using BLAS libraries like cuBLAS, CLBLas etc.
-   [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
-   [koboldcpp](https://github.com/LostRuins/koboldcpp) llama.cpp with a fancy UI, persistent stories, editing tools, memory etc. Supporting ggmlv3 and old ggml, CLBlast and llama, RWKV, GPT-NeoX, Pythia models
-   [Serge](https://github.com/nsarrazin/serge) chat interface based on llama.cpp for running Alpaca models. Entirely self-hosted, no API keys needed
-   [llama MPS](https://github.com/jankais3r/LLaMA_MPS) inference on Apple Silicon GPU using much lower power but is slightly slower than llama.cpp which uses CPU
-   [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp) Inference of HuggingFace's BLOOM-like models in pure C/C++
-   [RWKV.cpp](https://github.com/saharNooby/rwkv.cpp) CPU only port of BlinkDL/RWKV-LM to ggerganov/ggml. Supports FP32, FP16 and quantized INT4.
-   [RWKV Cuda](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda) a torchless, c++ rwkv implementation with 8bit quantization written in cuda
-   [secondbrain](https://github.com/juliooa/secondbrain) Multi-platform desktop app to download and run LLMs locally in your computer

gpt4all:
-   [gpt4all](https://github.com/nomic-ai/gpt4all) terminal and gui version to run local gpt-j models, [compiled binaries for win/osx/linux](https://gpt4all.io/index.html)
-   [gpt4all.zig](https://github.com/renerocksai/gpt4all.zig) terminal version of GPT4All
-   [gpt4all-chat](https://github.com/nomic-ai/gpt4all-chat) Cross platform desktop GUI for GPT4All  models (gpt-j)

others:
-   [Lit-llama](https://github.com/Lightning-AI/lit-llama) training, fine tuning and inference of llama
-   [Dalai](https://github.com/cocktailpeanut/dalai) LLaMA-based ChatGPT for single GPUs
-   [ChatLLaMA](https://github.com/juncongmoo/chatllama) LLaMA-based ChatGPT for single GPUs
-   [mlc-llm](https://github.com/mlc-ai/mlc-llm), run any LLM on any hardware (iPhones, Android, Win, Linux, Mac, WebGPU, Metal. NVidia, AMD)
-   [faraday.dev](https://faraday.dev/) Run open-source LLMs on your Win/Mac. Completely offline. Zero configuration.
-   [ChatALL](https://github.com/sunner/ChatALL) concurrently sends prompts to multiple LLM-based AI bots both local and APIs and displays the results
-   [pyllama](https://github.com/juncongmoo/pyllama) hacked version of LLaMA based on Meta's implementation, optimized for Single GPUs
-   [gmessage](https://github.com/drbh/gmessage) visually pleasing chatbot that uses a locally running LLM server and supports multiple themes, chat history search, text to speech, JSON file export, and OpenAI API compatible Python code
-   [selfhostedAI](https://github.com/josStorer/selfhostedAI) one-click deployment of RWKV, ChatGLM, llama.cpp models for substituting the openAI API to a locally hosted API 


## Web GUIs
### openAI
-   [TypingMind](https://www.typingmind.com/)
-   [Chatwithme.chat](https://www.chatwithme.chat/)
-   [enricoros/nextjs-chatgpt-app](https://github.com/enricoros/nextjs-chatgpt-app)
-   [no8081/chatgpt-demo](https://github.com/ddiu8081/chatgpt-demo)
-   [IPython-gpt](https://github.com/santiagobasulto/ipython-gpt) use chatGPT directly inside jupyter notebooks
-   [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) An open source ChatGPT UI

### Local LLMs
-   [Text Generation Webui](https://github.com/oobabooga/text-generation-webui) An all purpose UI to run LLMs of all sorts with optimizations ([running LLaMA-13b on 6GB VRAM](https://gist.github.com/rain-1/8cc12b4b334052a21af8029aa9c4fafc), [HN Thread](https://news.ycombinator.com/item?id=35937505))
-   [Text Generation Webui Ph0rk0z fork](https://github.com/Ph0rk0z/text-generation-webui-testing/) supporting all GPTQ versions and max context of 8192 instead of 4096 (because some models support longer context now)
-   [Alpaca-LoRa-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)
-   [chat petals](https://github.com/borzunov/chat.petals.ml) web app + HTTP and Websocket endpoints for BLOOM-176B inference with the Petals client
-   [Alpaca-Turbo](https://github.com/ViperX7/Alpaca-Turbo) Web UI to run alpaca model locally on Win/Mac/Linux
-   [FreedomGPT](https://github.com/ohmplatform/FreedomGPT) Web app that executes the FreedomGPT LLM locally
-   [HuggingChat](https://huggingface.co/chat) open source chat interface for transformer based LLMs by Huggingface
-   [openplayground](https://github.com/nat/openplayground) enables running LLM models on a laptop using a full UI, supporting various APIs and local HuggingFace cached models
-   [gpt4all Web UI](https://github.com/ParisNeo/gpt4all-ui) user friendly all-in-one interface, runs gpt_j, gptq, ggml and other model types
-   [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) Easy installation and running of RWKV Models, providing a local OpenAI API, GUI and custom CUDA kernel acceleration. Supports 2gb up to 32gb VRAM


## Voice Assistants
### openAI
-   [datafilik/GPT-Voice-Assistant](https://github.com/datafilik/GPT-Voice-Assistant)
-   [Abdallah-Ragab/VoiceGPT](https://github.com/Abdallah-Ragab/VoiceGPT)
-   [LlmKira/Openaibot](https://github.com/LlmKira/Openaibot)
-   [BarkingGPT](https://github.com/BudEcosystem/BarkingGPT) Audio2Audio by using Whisper+chatGPT+Bark
-  [gpt_chatbot](https://github.com/1nnovat1on/gpt_chatbot) Windows / elevenlabs TTS + pinecone long term memory
-  [gpt-voice-conversation-chatbot](https://github.com/Adri6336/gpt-voice-conversation-chatbot) using GPT3.5/4 API, elevenlab voices, google tts, session long term memory
-  [JARVIS-ChatGPT](https://github.com/gia-guar/JARVIS-ChatGPT) conversational assistant that uses OpenAI Whisper, OpenAI ChatGPT, and IBM Watson to provide quasi-real-time tips and opinions.
-   [ALFRED](https://github.com/masrad/ALFRED) LangChain Voice Assistant, powered by GPT-3.5-turbo, whisper, Bark, pyttsx3 and more
### Local LLMs
-   [bark TTS for oobabooga/text-generation-webui](https://github.com/wsippel/bark_tts) make your local LLM talk
-   [bark TTS for oobabooga/text-generation-webui](https://github.com/minemo/text-generation-webui-barktts) another implementation
-   [iris-llm](https://github.com/dkjroot/iris-llm/tree/prototypes) local voice chat agent


## Information retrieval
### openAI
-   [sqlchat](https://github.com/sqlchat/sqlchat) Use OpenAI GPT3/4 to chat with your database
-   [chat-with-github-repo](https://github.com/peterw/Chat-with-Github-Repo) which uses streamlit, gpt3.5-turbo and deep lake to answer questions about a git repo
-   [mpoon/gpt-repository-loader](https://github.com/mpoon/gpt-repository-loader) uses Git and GPT-4 to convert a repository into a text format for various tasks, such as code review or documentation generation.

### Local LLMs
-   [LlamaIndex](https://github.com/jerryjliu/llama_index) provides a central interface to connect your LLM's with external data
-   [Llama-lab](https://github.com/run-llama/llama-lab) home of llama_agi and auto_llama using LlamaIndex
-   [PrivateGPT](https://github.com/imartinez/privateGPT) a standalone question-answering system using LangChain, GPT4All, LlamaCpp and embeddings models to enable offline querying of documents
-   [Spyglass](https://github.com/spyglass-search/spyglass) tests an Alpaca integration for a self-hosted personal search app. Select the llama-rama feature branch. [Discussion on reddit](https://www.reddit.com/r/LocalLLaMA/comments/13key7p/a_little_demo_integration_the_alpaca_model_w_my/)
-   [local_llama](https://github.com/jlonge4/local_llama) chatting with your PDFs  offline. gpt_chatwithPDF alternative with the ultimate goal of using llama instead of chatGPT
-   [Sidekick](https://github.com/ai-sidekick/sidekick) Information retrieval for LLMs
-   [DB-GPT](https://github.com/csunny/DB-GPT) SQL generation, private domain Q&A, data processing, unified vector storage/indexing, and support for various plugins and LLMs
-   [localGPT](https://github.com/PromtEngineer/localGPT) a privateGPT inspired document question-answering solution using GPU instead of CPU acceleration and InstructorEmbeddings, which perform better [according to leaderboards](https://huggingface.co/spaces/mteb/leaderboard) instead of LlamaEmbeddings
-   [LocalDocs](https://docs.gpt4all.io/gpt4all_chat.html#how-localdocs-works) plugin for GPT4All
-   [annoy_ltm](github.com/YenRaven/annoy_ltm) extension to add long term memory to chatbots using a nearest neighbor vector DB for memory retrieval
-   [ChatDocs](https://github.com/marella/chatdocs) PrivateGPT + Web UI + GPU Support + ggml, transformers, webui


### Model Agnostic
-   [Paper QA](https://github.com/whitead/paper-qa) LLM Chain for answering questions from documents with citations, using OpenAI Embeddings or local llama.cpp, langchain and FAISS Vector DB
-   [BriefGPT](https://github.com/e-johnstonn/BriefGPT) document summarization and querying using OpenAI' and locally run LLM's using LlamaCpp or GPT4ALL, and embeddings stored as a FAISS index, built using Langchain.

## Browser Extensions
### openAI
-   [sider](https://chrome.google.com/webstore/detail/sider-chatgpt-sidebar-gpt/difoiogjjojoaoomphldepapgpbgkhkb) chrome side-bar for chatGPT and OpenAI API supporting custom prompts and text highlighting
-   [chathub-dev/chathub](https://github.com/chathub-dev/chathub)
-   [Glarity](https://github.com/sparticleinc/chatgpt-google-summary-extension) open-source chrome extension to write summaries for various websites including custom ones and YouTube videos. Extensible
-   [superpower-chatgpt](https://github.com/saeedezzati/superpower-chatgpt) chrome  extension / firefox addon to add missing features like Folders, Search, and Community Prompts to ChatGPT

### Local LLMs
-   [chatGPTBox](https://github.com/josStorer/chatGPTBox) add useful LLM chat-boxes to github and other websites, supporting self-hosted model (RWKV, llama.cpp, ChatGLM)


## Agents / Automatic GPT
### openAI
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
-   [Chrome-GPT](https://github.com/richardyc/Chrome-GPT)  AutoGPT agent employing Langchain and Selenium to interact with a Chrome browser session, enabling Google search, webpage description, element interaction, and form input
-   [autolang](https://github.com/alvarosevilla95/autolang) Another take on BabyAGI, focused on workflows that complete. Powered by langchain.
-   [ai-legion](https://github.com/eumemic/ai-legion) A framework for autonomous agents who can work together to accomplish tasks.
-   [generativeAgent_LLM](https://github.com/QuangBK/generativeAgent_LLM) Generative Agents with Guidance, Langchain, and local LLMs, implementation of the "Generative Agents: Interactive Simulacra of Human Behavior" [paper](https://arxiv.org/pdf/2304.03442.pdf), [blogpost](https://betterprogramming.pub/implement-generative-agent-with-local-llm-guidance-and-langchain-full-features-fa57655f3de1)

### Local LLMs
-   [Auto Vicuna Butler](https://github.com/NiaSchim/auto-vicuna-butler) Baby-AGI fork / AutoGPT alternative to run with local LLMs
-   [BabyAGI](https://github.com/yoheinakajima/babyagi) AI-Powered Task Management for OpenAI + Pinecone or Llama.cpp
-   [Agent-LLM](https://github.com/Josh-XT/Agent-LLM) Webapp to control an agent-based Auto-GPT alternative, supporting GPT4, Kobold, llama.cpp, FastChat, Bard, Oobabooga textgen
-   [auto-llama-cpp](https://github.com/rhohndorf/Auto-Llama-cpp) fork of Auto-GPT with added support for locally running llama models through llama.cpp
-   [AgentOoba](https://github.com/flurb18/AgentOoba) autonomous AI agent extension for Oobabooga's web ui
-   [RecurrentGPT](https://github.com/aiwaves-cn/RecurrentGPT/tree/main) Interactive Generation of (Arbitrarily) Long Text. Uses LSTM, prompt-engineered recurrence, maintains short and long-term memories, and updates these using semantic search and paragraph generation.


## Multi Modal
-   [Alpaca-Turbo | Web UI to run alpaca model locally on Win/Mac/Linux](https://github.com/ViperX7/Alpaca-Turbo)
-   [FreedomGPT | Web app that executes the FreedomGPT LLM locally](https://github.com/ohmplatform/FreedomGPT)
-   [huggingGPT / JARVIS](https://github.com/microsoft/JARVIS) Connects LLMs with huggingface specialized models
-   [Langchain-huggingGPT](https://github.com/camille-vanhoffelen/langchain-huggingGPT) reimplementation of HuggingGPT using langchain
-   [OpenAGI](https://github.com/agiresearch/openagi) AGI research platform, solves multi step tasks with RLTF and supports complex model chains
-   [ViperGPT](https://github.com/cvlab-columbia/viper) implementation for visual inference and reasoning with openAPI
-   [TaskMatrix](https://github.com/microsoft/visual-chatgpt) former visual-chatgpt connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting.
-   [PandaGPT](https://github.com/yxuansu/PandaGPT) combines ImageBind and Vicuna to understand and combine multimodal inputs from text, image, audio, depth, thermal, and IMU.

## Code generation
-   [FauxPilot](https://github.com/fauxpilot/fauxpilot) open source Copilot alternative using Triton Inference Server
-   [Turbopilot](https://github.com/ravenscroftj/turbopilot) open source LLM code completion engine and Copilot alternative
-   [Tabby](https://github.com/TabbyML/tabby) Self hosted Github Copilot alternative
-   [starcoder.cpp](https://github.com/bigcode-project/starcoder.cpp)
-   [GPTQ-for-SantaCoder](https://github.com/mayank31398/GPTQ-for-SantaCoder) 4bit quantization for SantaCoder
-   [supercharger](https://github.com/catid/supercharger) Write Software + unit tests for you, based on Baize-30B 8bit, using model parallelism
-   [Autodoc](https://github.com/context-labs/autodoc) toolkit that auto-generates codebase documentation using GPT-4 or Alpaca, and can be installed in a git repository in about 5 minutes.
-   [smol-ai developer](https://github.com/smol-ai/developer) a personal junior developer that scaffolds an entire codebase with a human-centric and coherent whole program synthesis approach using <200 lines of Python and Prompts.
-   [locai](https://github.com/Maykeye/locai) kobold/oobabooga -compatible api for vscode
-   [oasis](https://github.com/paolorechia/oasis) local LLaMA models in VSCode

## Libraries and Wrappers
### openAI
-   [acheong08/ChatGPT Python](https://github.com/acheong08/ChatGPT) reverse engineerded chatGPT API
-   [gpt4free](https://github.com/xtekky/gpt4free) Use reverse engineered GPT3.5/4 APIs of other website's APIs
-   [GPTCache](https://github.com/zilliztech/GPTCache), serve cached results based on embeddings in a vector DB, before querying the OpenAI API.
-   [kitt](https://github.com/livekit-examples/kitt) TTS + GPT4 + STT to create a conference call audio bot
-   [Marvin](https://github.com/prefecthq/marvin) simplifies AI integration in software development with easy creation of AI functions and bots managed through a conversational interface
-   [chatgpt.js](https://github.com/chatgptjs/chatgpt.js) client-side JavaScript library for ChatGPT
-   [ChatGPT-Bridge](https://github.com/improveTheWorld/ChatGPT-Bridge) use chatGPT plus' GPT-4 as a local API
-   [Powerpointer](https://github.com/CyberTimon/powerpointer) connects to openAPI GPT3.5 and creates a powerpoint out of your content
-   [EdgeGPT](https://github.com/acheong08/EdgeGPT) Reverse engineered API of Microsoft's Bing Chat using Edge browser


### Local LLMs
-   [FastLLaMA Python wrapper for llama.cpp](https://github.com/PotatoSpudowski/fastLLaMa)
-   [WebGPT](https://github.com/0hq/WebGPT) Inference in pure javascript
-   [TokenHawk](https://github.com/kayvr/token-hawk) performs hand-written LLaMA inference using WebGPU, utilizing th.cpp, th-llama.cpp, and th-llama-loader.cpp, with minimal dependencies
-   [WasmGPT](https://github.com/lxe/ggml/tree/wasm-demo) ChatGPT-like chatbot in browser using ggml and emscripten
-   [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) easy-to-use model GPTQ quantization package with user-friendly CLI
-   [gpt-llama.cpp](https://github.com/keldenl/gpt-llama.cpp) Replace OpenAi's GPT APIs with llama.cpp's supported models locally
-   [llama-node](https://github.com/Atome-FE/llama-node) JS client library for llama (or llama based) LLMs built on top of llama-rs and llama.cpp.
-   [TALIS](https://github.com/Dhaladom/TALIS) serves a LLaMA-65b API, optimized for speed utilizing dual RTX 3090/4090 GPUs on Linux
-   [Powerpointer-For-Local-LLMs](https://github.com/CyberTimon/Powerpointer-For-Local-LLMs) connects to oobabooga's API and creates a powerpoint out of your content
-   [OpenChatKit](https://github.com/togethercomputer/OpenChatKit) open-source project that provides a base to create both specialized and general purpose chatbots and extensible retrieval system, using GPT-NeoXT-Chat-Base-20B as a base model
-   [webgpu-torch](https://github.com/praeclarum/webgpu-torch) Tensor computation with WebGPU acceleration
-   [llama-api-server](https://github.com/iaalm/llama-api-server) that uses llama.cpp and emulates an openAI API
-   [CTransformers](https://github.com/marella/ctransformers) python bindings for transformer models in C/C++ using GGML library, supporting GPT-2/J/NeoX, StableLM, LLaMA, MPT, Dollyv2, StarCoder
-   [basaran](https://github.com/hyperonym/basaran) GUI and API as a drop-in replacement of the OpenAI text completion API. Broad HF eco system support (not only llama)
-   [CodeTF](https://github.com/salesforce/CodeTF) one-stop Python transformer-based library for code LLMs and code intelligence, training and inferencing on code summarization, translation, code generation

### Model agnostic
-   [LangChain | framework for developing LLM applications](https://github.com/hwchase17/langchain) ([example](https://www.youtube.com/watch?v=iRJ4uab_NIg&t=588s), [paolorechia/learn-langchain with vicuna and GPTQ 4 bit support](https://github.com/paolorechia/learn-langchain))
-   [LangFlow](https://github.com/logspace-ai/langflow) GUI for Langchain
-   [Toolformer implementation](https://github.com/lucidrains/toolformer-pytorch) Allows LLMs to use Tools
-   [megabots](https://github.com/momegas/megabots) to create LLM bots by providing Q&A, document retrieval, vector DBs, FastAPI, Gradio UI, GPTCache, guardrails, whisper, supports OpenAI API (local LLMs planned)
-   [Jsonformer](https://github.com/1rgs/jsonformer): Generate Structured JSON from Language Models by handling JSON synthax, and letting LLM just output the values
-   [gorilla](https://github.com/ShishirPatil/gorilla) Enables LLMs to use tools by semantically and syntactically correctly invoking APIs. Reduces hallucination, custom trained model [weights](https://huggingface.co/TheBloke/gorilla-7B-fp16) based on llama-7b




## Fine Tuning & Training
- [simple llama finetuner](https://github.com/lxe/simple-llama-finetuner)
- [LLaMA-LoRA Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner)
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
- [StackLLaMA Fine-Tuning Guide by huggingface](https://huggingface.co/blog/stackllama)
- [xTuring](https://github.com/stochasticai/xturing) LLM finetuning pipeline supporting LoRa & 4bit
- [Microsoft DeepSpeed Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md)
- [How to train your LLMs](https://blog.replit.com/llm-training)
- [H2O LLM Studio | Framework and no-code GUI for fine tuning SOTA LLMs](https://github.com/h2oai/h2o-llmstudio)
- [Implementation of LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter), to fine tune instructions within hours
- [Hivemind](https://github.com/learning-at-home/hivemind) Training at home
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) a llama, pythia, cerebras training environment optimized for Runpod supporting qlora, 4bit, flash attention, xformers
- [LMFlow](https://github.com/OptimalScale/LMFlow) toolbox for finetuning, designed to be user-friendly, speedy, and reliable
- [qlora](https://github.com/artidoro/qlora) uses bitsandbytes quantization and PEFT and transformers for efficient finetuning of quantized LLMs
- [GPTQlora](https://github.com/qwopqwop200/gptqlora) Efficient Finetuning of Quantized LLMs with GPTQ QLoRA and AutoGPTQ for quantization

## Frameworks
- [Vicuna FastChat](https://github.com/lm-sys/FastChat)
- [SynapseML](https://github.com/microsoft/SynapseML) (previously known as MMLSpark),an open-source library that simplifies the creation of massively scalable machine learning (ML) pipelines
- [Microsoft guidance](https://github.com/microsoft/guidance) efficient Framework for Enhancing Control and Structure in Modern Language Model Interactions. [Demo project](https://github.com/paolorechia/local-guidance) by paolorechia for local text-generation-webui. [reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/13kzubz/i_made_a_simple_agent_demo_with_guidance_and/). [guidance fork](https://github.com/Maximilian-Winter/guidance) and [llama-cpp-python fork](https://github.com/Maximilian-Winter/llama-cpp-python) how-to on [reddit](https://www.reddit.com/r/LocalLLaMA/comments/13magac/hi_community_i_created_a_fork_of_microsofts/)
- [Microsoft semantic-kernel](https://github.com/microsoft/semantic-kernel) a lightweight SDK enabling integration of AI Large Language Models (LLMs) with conventional programming languages
- [Colossal-AI](https://github.com/hpcaitech/ColossalAI) unified deep learning system that provides a collection of parallel components for distributed deep learning models. Provides data parallelism, pipeline parallelism, and tensor parallelism

# Resources
## Data sets
- [Alpaca-lora](https://github.com/tloen/alpaca-lora) instruction finetuned using Low Rank Adaption
- [codealpaca](https://github.com/sahil280114/codealpaca) Instruction training data set for code generation
- [LAION AI / Open-Assistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) (https://github.com/LAION-AI/Open-Assistant / https://projects.laion.ai/Open-Assistant/ / https://open-assistant.io)
- [ShareGPT pre-cleaned, English only](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) "unfiltered," and 2048 token split version of the ShareGPT dataset ready for finetuning
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
- [Alpaca Chain of Thought](https://github.com/PhoebusSi/Alpaca-CoT) fine tuning dataset for EN and CN
- [PRESTO](https://ai.googleblog.com/2023/03/presto-multilingual-dataset-for-parsing.html) [paper](https://arxiv.org/pdf/2303.08954.pdf) Multilingual dataset for parsing realistic task-oriented dialogues by Google & University of Rochester, California, Santa Barbara, Columbia
- [RedPajama](https://www.together.xyz/blog/redpajama) Dataset and model similar to LLaMA but truly open source and ready for commercial use. [hf](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- [BigCode The Stack](https://huggingface.co/datasets/bigcode/the-stack)
- [open-instruct-v1](https://huggingface.co/datasets/hakurei/open-instruct-v1)
- [awesome-instruction-dataset](https://github.com/yaodongC/awesome-instruction-dataset) list of instruction datasets by yadongC
- [The Embedding Archives](https://txt.cohere.com/embedding-archives-wikipedia/) Millions of Wikipedia Article Embeddings in multiple languages
- [Rereplit-finetuned-v1-3b & replit-code-v1-3b](https://twitter.com/Replit/status/1651344186715803648) outperforming all coding OSS models, gets released soon
- [alpaca_evol_instruct_70k](https://huggingface.co/datasets/victor123/evol_instruct_70k) an instruction-following dataset created using [Evol-Instruct](https://github.com/nlpxucan/evol-instruct), used to fine-tune [WizardLM](https://github.com/nlpxucan/WizardLM)
- [gpt4tools_71k.json](https://github.com/StevenGrove/GPT4Tools#Dataset) from GPT4Tools paper, having 71k instruction-following examples for sound/visual/text instructions
- [WizardVicuna 70k dataset](https://huggingface.co/datasets/junelee/wizard_vicuna_70k) used to fine tune [WizardVicuna](https://github.com/melodysdreamj/WizardVicunaLM)
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers)
- [airoboros uncensored](https://huggingface.co/datasets/jondurbin/airoboros-uncensored)
- [CoT collection](https://github.com/kaistAI/CoT-Collection), [paper](https://arxiv.org/abs/2305.14045)
- [airoboros-gpt4](https://huggingface.co/datasets/jondurbin/airoboros-gpt4) fine-tuning dataset optimized for trivia, math, coding, closed context question answering, multiple choice, writing

## Research
- [LLM Model Cards](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs)
- [GPTs are GPTs: An early look at the labor market impact potential of LLMs](https://arxiv.org/abs/2303.10130)
- [ViperGPT](https://viper.cs.columbia.edu/) Visual Inference via Python Execution for reasoning
- [Emergent Abilities of LLMs ](https://openreview.net/forum?id=yzkSU5zdwD), [blog post](https://www.jasonwei.net/blog/emergence)
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
- [Hyena](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena) a subquadratic-time layer that has the potential to significantly increase context length in sequence models, using a combination of long convolutions and gating. [Long Convs and Hyena implementations](https://github.com/hazyresearch/safari)
- [FastServe](https://arxiv.org/abs/2305.05920) an efficient distributed inference serving system for LLMs that minimizes job completion time using preemptive scheduling and efficient GPU memory management, built on NVIDIA FasterTransformer.
- [FrugalGPT](https://arxiv.org/abs/2305.05176) is a model that uses LLM cascade to optimize the performance and cost-efficiency of LLMs like GPT-4.
- [Landmark Attention](https://arxiv.org/abs/2305.16300) LLaMa 7B with 32k tokens. [Code](https://github.com/epfml/landmark-attention), llama7b diff [weights](https://huggingface.co/epfml/landmark-attention-llama7b-wdiff), merged llama7b [weights](https://huggingface.co/TheBloke/landmark-attention-llama7b-fp16)
- [QLORA](https://arxiv.org/abs/2305.14314) Efficient Finetuning of Quantized LLMs
- [Tree of Thoughts (ToT)](https://arxiv.org/abs/2305.10601) Enables exploration over text, improves strategic decision-making in language models. [Code](https://github.com/ysymyth/tree-of-thought-llm). Example [implementation](https://github.com/kyegomez/tree-of-thoughts), [discussion](https://github.com/kyegomez/tree-of-thoughts/pull/8)
- [MEGABYTE](https://arxiv.org/abs/2305.07185) Efficient multiscale decoder architecture for long-sequence modeling.
- [PandaGPT](http://arxiv.org/abs/2305.16355):  [project page](https://panda-gpt.github.io/), [code](https://github.com/yxuansu/PandaGPT), [model](https://huggingface.co/openllmplayground/pandagpt_13b_max_len_400) combines ImageBind and Vicuna to understand and combine multimodal inputs from text, image, audio, depth, thermal, and IMU.
- [LIMA](https://arxiv.org/abs/2305.11206) Less Is More for Alignment. Shows fine-tuning with 1000 carefully curated prompts without reinforcement learning can outperforms GPT-4 in many cases
- [Gorilla](https://arxiv.org/abs/2305.15334) a finetuned LLaMA-based model that surpasses GPT-4 in writing API calls and reduces hallucination. [project](https://gorilla.cs.berkeley.edu/), [code](https://github.com/ShishirPatil/gorilla)
- [Voyager](https://arxiv.org/abs/2305.16291) Open-Ended Embodied Minecraft Agent using LLMs, [project](https://voyager.minedojo.org/), [code](https://github.com/MineDojo/Voyager)
- [BigTrans](https://huggingface.co/papers/2305.18098) llama adapted to multilingual translation over 100 languages, outperforming chatGPT in 8 language-pairs
- [BPT](https://arxiv.org/pdf/2305.19370.pdf) memory-efficient approach to processing long input sequences in Transformers
- [Lion](https://arxiv.org/abs/2305.12870) efficiently transfers knowledge from a closed-source LLM to an open-source student model
- [Undetectable Watermarks for Language Models](https://eprint.iacr.org/2023/763) using one-way functions
- [ALiBi](https://arxiv.org/abs/2108.12409) Train Short Test Long. Attention with Linear Biases Enables Input Length Extrapolation. [code](https://github.com/ofirpress/attention_with_linear_biases)
- [The Curse of Recursion](https://arxiv.org/abs/2305.17493): Training on Generated Data Makes Models Forget
- [Brainformers](https://arxiv.org/abs/2306.00008) a complex block for natural language processing that outperforms state-of-the-art Transformers in efficiency and quality

# Other awesome resources
- [LLM Worksheet](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=741531996) by [randomfoo2](https://www.reddit.com/r/LocalAI/comments/12smsy9/list_of_public_foundational_models_fine_tunes/)
- [The full story of LLMs](https://www.assemblyai.com/blog/the-full-story-of-large-language-models-and-rlhf/)
- [Brief history of llama models](https://agi-sphere.com/llama-models/)
- [A timeline of transformer models](https://ai.v-gar.de/ml/transformer/timeline/)
- [Every front-end GUI client for ChatGPT API](https://github.com/billmei/every-chatgpt-gui)
- [LLMSurvey](https://github.com/rucaibox/llmsurvey) a collection of papers and resources including an LLM timeline
- [rentry.org/lmg_models](https://rentry.org/lmg_models) a list of llama derrivates and models
- [Timeline of AI and language models](https://lifearchitect.ai/timeline/) and [Model Comparison Sheet](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs/edit#gid=1158069878) by Dr. Alan D. Thompson
- [Brex's Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering) an evolving manual providing historical context, strategies, guidelines, and safety recommendations for building programmatic systems on OpenAI's GPT-4
- [LLMs Practical Guide](https://github.com/Mooler0410/LLMsPracticalGuide) actively curated collection of a timeline and guides for LLMs, providing a historical context and restrictions based on [this](https://arxiv.org/abs/2304.13712) paper and community contributions
- [LLMSurvey](https://github.com/RUCAIBox/LLMSurvey) based on [this](https://arxiv.org/abs/2303.18223) paper, builds a collection of further papers and resources related to LLMs including a timeline
- [LLaMAindex](https://medium.com/llamaindex-blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec) can now use Document Summary Index for better QA performance compared to vectorDBs
- [ossinsight.io chat-gpt-apps](https://ossinsight.io/collections/chat-gpt-apps/) Updated list of top chatGPT related repositories

## Product Showcases
- [Opinionate.io AI Debating AI](https://opinionate.io/)
- [phind.com](phind.com) Developer Search Engine
- [Voice Q&A Assistant](https://github.com/hackingthemarkets/qa-assistant-eleven-labs-voice-cloning) using ChatGPT API, Embeddings, Gradio, Eleven Labs and Whisper
- [chatpdf](https://www.chatpdf.com/), Q&A for PDFs

## Optimization
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
- https://github.com/HazyResearch/flash-attention
- https://github.com/stochasticai/x-stable-diffusion
- [ExLlama](https://github.com/turboderp/exllama) a more memory-efficient rewrite of the HF transformers implementation of Llama for use with quantized weights. By [ReturningTarzan](https://www.reddit.com/r/LocalLLaMA/comments/13vm7tx/can_an_rtx_3090_run_a_30b_4bit_model_with_webui/jm6wmu0/)
- [tokenmonster](https://github.com/alasdairforsythe/tokenmonster) ungreedy tokenizer increases inference speed and context-length by 35% for pre-training on new LLMs

## Benchmarking
### Leaderboards
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) by HuggingFaceH4
- [LMSys Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard), [blogpost](https://lmsys.org/blog/2023-05-03-arena/) is an anonymous benchmark platform for LLMs that features randomized battles in a crowdsourced manner
- [Current best choices](https://old.reddit.com/r/LocalLLaMA/wiki/models#wiki_current_best_choices) on LocalLLaMA reddit
- [LLM Logic Tests](https://docs.google.com/spreadsheets/d/1NgHDxbVWJFolq8bLvLkuPWKC7i_R6I6W/edit#gid=2011456595) by [YearZero on reddit/localllama](https://www.reddit.com/r/LocalLLaMA/comments/13636h5/updated_riddlecleverness_comparison_of_popular/)
- [paperswithcode](https://paperswithcode.com/) has LLM SOTA leaderboards, but usually just for foundation models
- [Can AI code](https://github.com/the-crypt-keeper/can-ai-code) a self-evaluating interview for AI coding models.
- [Gotzmann LLM Score v2](https://docs.google.com/spreadsheets/d/1ikqqIaptv2P4_15Ytzro46YysCldKY7Ub2wcX5H1jCQ/edit#gid=0) by [Gatzuma on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/13wvd0j/llm_score_v2_modern_models_tested_by_human/)
- [Aviary Explorer](https://aviary.anyscale.com/) open source utility to compare leading OSS LLMs and see votes, pricing per token etc.


### Benchmark Suites
- [Big-bench](https://github.com/google/BIG-bench) a collaborative benchmark featuring over 200 tasks for evaluating the capabilities of llms
- [Pythia](https://github.com/EleutherAI/pythia) interpretability analysis for autoregressive transformers during training
- 
## AI DevOps
- https://www.steamship.com/

## Databases for ML
- [Pinecone](https://www.pinecone.io/) proprietary vector search for semantic search, recommendations and information retrieval
- [FAISS](https://github.com/facebookresearch/faiss) Library for Efficient Similarity Search and Clustering using vectors
- [Weaviate](https://github.com/weaviate/weaviate) open source vector DB for services like OpenAI, HF etc for text, image, Q&A etc.
- [vespa.ai](https://vespa.ai/) one of the only scalable vector DBs that supports multiple vectors per schema field
- [LanceDB](https://github.com/lancedb/lancedb) free open-source serverless vector DB with support for langchain, llamaindex and multi-modal data
- [Deeplake](https://github.com/activeloopai/deeplake) Vector Database for audio, text, vectors, video
- [milvus](https://github.com/milvus-io/milvus) open-source cloud-native vector DB focusing on embedding vectors converted from unstructured data
