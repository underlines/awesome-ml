[🏠Home](README.md)

## Open LLM Models List

Due to projects like [Explore the LLMs](https://llm.extractum.io/) specializing in model indexing, the custom list has been removed.


## Noteworthy

- [Cerebras GPT-13b](https://huggingface.co/cerebras) ([release notes](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/))
- [LAION OpenFlamingo | Multi Modal Model and training architecture](https://github.com/mlfoundations/open_flamingo)
- [GeoV/GeoV-9b](https://huggingface.co/GeoV/GeoV-9b) - 9B parameter, in-progress training to 300B tokens (33:1)
- [RWKV: Parallelizable RNN with Transformer-level LLM Performance](https://github.com/BlinkDL/RWKV-LM)
- [CodeGeeX 13B | Multi Language Code Generation Model](https://huggingface.co/spaces/THUDM/CodeGeeX)
- [BigCode | Open Scientific collaboration to train a coding LLM](https://huggingface.co/bigcode)
- [MOSS by Fudan University](https://github.com/OpenLMLab/MOSS) a 16b Chinese/English custom foundational model with additional models fine tuned on sft and plugin usage
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) Multimodal finetuned model for visual/language tasks
- [Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT) multi-modal visual/language chatbot, using llama with custom LoRA weights and openflamingo-9B.
- [Visual-med-alpaca](https://github.com/cambridgeltl/visual-med-alpaca) fine-tuning llama-7b on self instruct for the biomedical domain. Models locked behind a request form.
- [replit-code](https://huggingface.co/replit/) focused on Code Completion. The model has been trained on a subset of the [Stack Dedup v1.2](https://arxiv.org/abs/2211.15533) dataset.
- [VPGTrans](https://vpgtrans.github.io/) Transfer Visual Prompt Generator across LLMs and the VL-Vicuna model is a novel VL-LLM. [Paper](https://arxiv.org/abs/2305.01278), [code](https://github.com/VPGTrans/VPGTrans)
- [salesforce/CodeT5](https://github.com/salesforce/codet5) code assistant, has released their [codet5+ 16b](https://huggingface.co/Salesforce/codet5p-16b) and other model sizes
- [baichuan-7b](https://github.com/baichuan-inc/baichuan-7B) Baichuan Intelligent Technology developed baichuan-7B, an open-source language model with 7 billion parameters trained on 1.2 trillion tokens. Supporting Chinese and English, it achieves top performance on authoritative benchmarks (C-EVAL, MMLU)
- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) v2 of the GLM 6B open bilingual EN/CN model
- [sqlcoder](https://github.com/defog-ai/sqlcoder) 15B parameter model that outperforms gpt-3.5-turbo for natural language to SQL generation tasks
- [CodeShell](https://github.com/WisdomShell/codeshell/blob/main/README_EN.md) code LLM with 7b parameters trained on 500b tokens, context length of 8k outperforming CodeLlama and Starcoder on humaneval, [weights](https://huggingface.co/WisdomShell/CodeShell)
- [SauerkrautLM-13B-v1](https://huggingface.co/VAGOsolutions/SauerkrautLM-13b-v1) fine tuned llama-2 13b on a mix of German data augmentation and translations, [SauerkrautLM-7b-v1-mistral](https://huggingface.co/VAGOsolutions/SauerkrautLM-7b-v1-mistral) German SauerkrautLM-7b fine-tuned using QLoRA on 1 A100 80GB with Axolotl
- [em_german_leo_mistral](https://huggingface.co/jphme/em_german_leo_mistral) LeoLM Mistral fine tune of [LeoLM](https://huggingface.co/LeoLM/leo-hessianai-13b) with german instructions
- [leo-hessianai-13b-chat-bilingual](https://huggingface.co/LeoLM/leo-hessianai-13b-chat-bilingual) based on llama-2 13b is a fine tune of the base [leo-hessianai-13b](https://huggingface.co/LeoLM/leo-hessianai-13b) for chat
- [WizardMath-70B-V1.0](https://huggingface.co/WizardLM/WizardMath-70B-V1.0) SOTA Mathematical Reasoning
- [Mistral-7B-german-assistant-v3](https://huggingface.co/flozi00/Mistral-7B-german-assistant-v3) finetuned version for german instructions and conversations in style of Alpaca. "### Assistant:" "### User:", trained with a context length of 8k tokens. The dataset used is deduplicated and cleaned, with no codes inside. The focus is on instruction following and conversational tasks
- [HelixNet](https://huggingface.co/migtissera/HelixNet) Mixture of Experts with 3 Mistral-7B, [LoRA](https://huggingface.co/rhysjones/HelixNet-LMoE-Actor), [HelixNet-LMoE](https://huggingface.co/rhysjones/HelixNet-LMoE-6.0bpw-h6-exl2) optimized version
- [llmware RAG models](https://huggingface.co/llmware) small LLMs and sentence transformer embedding models specifically fine-tuned for RAG workflows
- [openchat](https://github.com/imoneoi/openchat) Advancing Open-source Language Models with Mixed-Quality Data
- [deepseek-coder](https://github.com/deepseek-ai/DeepSeek-Coder) code language models, trained on 2T tokens, 87% code 13% English / Chinese, up to 33B with 16K context size achieving SOTA performance on coding benchmarks
- [Poro](https://huggingface.co/LumiOpen/Poro-34B) SiloGen model checkpoints of a family of multilingual open source LLMs covering all official European languages and code, [news](https://joinup.ec.europa.eu/collection/open-source-observatory-osor/news/new-open-source-ai-model-poro-challenges-french-mistral)
- [Mixtral of experts](https://mistral.ai/news/mixtral-of-experts/) A high quality Sparse Mixture-of-Experts.
- [meditron](https://github.com/epfLLM/meditron) 7B and 70B Llama2 based LLM fine tuning adapted for the medical domain
- [SeaLLM](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2) multilingual LLM for Southeast Asian (SEA) languages 🇬🇧 🇨🇳 🇻🇳 🇮🇩 🇹🇭 🇲🇾 🇰🇭 🇱🇦 🇲🇲 🇵🇭
- [seamlessM4T v2](https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2) Multimodal Audio and Text Translation between many languages
- [aya-101](https://huggingface.co/CohereForAI/aya-101) 13b model fine tuned open acess multilingual LLM from Cohere For AI
- [SLIM Model Family](https://huggingface.co/llmware) Small Specialized Function-Calling Models for Multi-Step Automation, focused on enterprise RAG workflows
- [Smaug-72B](https://huggingface.co/abacusai/Smaug-72B-v0.1) Based on Qwen-72B and MoMo-72B-Lora then finetuned by Abacus.AI, is the best performing Open LLM on the HF leaderboard by Feb-2024
- [AI21 Jamba](https://huggingface.co/ai21labs/Jamba-v0.1) production-grade Mamba-based hybrid SSM-Transformer Model licensed under Apache 2.0 with 256K context and 52B MoE at 12B each
- [command-r](https://www.maginative.com/article/cohere-launches-command-r-scalable-ai-model-for-enterprise-rag-and-tool-use/) 35B optimized for retrieval augmented generation (RAG) and tool use supporting Embed and Rerank methodology. [model weights](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [StarCoder2](https://huggingface.co/bigcode/starcoder2-15b) 15B, 7B and 3B code completion models trained on The Stack v2
- [command-r-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) a 104B model with highly advanced capabilities including RAG and tool use for English, French, Spanish, Italian, German, Brazilian Portuguese, Japanese, Korean, Arabic, and Simplified Chinese
- [DBRX](https://huggingface.co/databricks/dbrx-base) base and instruct MoE models from databricks with 132B total parameters and a larger number of smaller experts supporting RoPE and 32K context size
- [grok-1](https://huggingface.co/xai-org/grok-1) 314b MoE model by xAI
- [Mixtral-8x22B-v0.1](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) Sparse MoE model with 176B total and 44B active parameters, 65k context size
- [aiXcoder](https://huggingface.co/aiXcoder/aixcoder-7b-base) 7B Code LLM for code completion, comprehension, generation
- [WizardLM-2-7B](https://huggingface.co/microsoft/WizardLM-2-7B) Microsoft's WizardLM 2 7B, release for 70B coming up [backup0](https://huggingface.co/lucyknada/microsoft_WizardLM-2-7B)
- [WizardLM-2-8x22B](https://huggingface.co/alpindale/WizardLM-2-8x22B) Microsoft's WizardLM 2 8x22B beating gpt-4-0314 on MT-Bench
- [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) an instruct fine-tuned version of the Mixtral-8x22B-v0.1
- [wavecoder-ultra-6.7b](https://huggingface.co/microsoft/wavecoder-ultra-6.7b) covering four general code-related tasks: code generation, code summary, code translation, and code repair
- [GemMoE](https://huggingface.co/Crystalcareai/GemMoE-Base-Random) An 8x8 Mixture Of Experts based on Gemma
- [Granite](https://huggingface.co/ibm-granite) family of Code Models from IBM with 3b, 8b, 20b, 34b, base and instruct models for code completion and chat
- [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2#2-model-downloads) 21B Strong, Economical, and Efficient Mixture-of-Experts Language Model
- [Yuan2-M32](https://huggingface.co/IEITYuan/Yuan2-M32-hf) Mixture of Experts with Attention Router, 32 Experts, 2 Active, TOtal 40B parameters, 3.7B active and max length of 16K
- [CodeStral-22B](https://huggingface.co/mistralai/Codestral-22B-v0.1) Coding model trained on 80+ languages with instruct and Fill in the Middle tasks, 32k max context
- [Mistral-7b-instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) with function calling, new tokenizer and 32k max context
- [Aya-23](https://huggingface.co/CohereForAI/aya-23-35B) 8B and 35B instruction tuned multi lingual model focusing on 23 languages
- [Mamba-Codestral](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1) by mistral based on the Mamba2 architecture performing on par with SOTA transformer based code models
- [CodeGeeX4](https://huggingface.co/THUDM/codegeex4-all-9b) 9B multilingual code generation model for chat and instruct with a 128k context length
- [Mistral Nemo](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) a 12B model by mistral and nvidia offering 128k context window offered as instruct and base models
- [Nuextract](https://huggingface.co/numind/NuExtract) is a structure extraction model based on phi-3-mini, allowing to instruct based on a json template that the model fills from unstructured text provided
- [Llama-3.1](https://ai.meta.com/blog/meta-llama-3-1/) Metas most advanced model providing 8b, 70b and 405b base and instruction tuned models and 128k context window with on par quality of current SOTA closed source models
- [Mistral-Large](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) a 123B sized model beating llama-3.1 and gpt-4o in several categories with a focus on multilinguality, coding, agentic tasks and reasoning.
- [InternLM2.5](https://huggingface.co/internlm/internlm2_5-7b-chat) 7B base and chat models focusing reasoning, math and tool use and 1M context window
- [Yi-1.5]([https://huggingface.co/01-ai/Yi-9B](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)) 9b model focusing on multilingual text understanding, available as 9B and 34B variants
- [Phi](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) Microsoft's small language and vision models with small and medium parameter sizes, short and long context lengths and great performance
- [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) English and Chinese models from 0.5b, 1.5b, 7b, and 72b sizes with great performance and 128k context windows for the 7 and 72b models
- [codeqwen1.5](https://huggingface.co/Qwen/CodeQwen1.5-7B) base and chat models with 7B parameters and good quality
- [grantie](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330) IBMs code models available in 3b, 8b, 20b size as base and instruct variants with up to 128k context size
- [codegemma](https://huggingface.co/google/codegemma-7b) google's coding models from 2b base, 7b base and 7b instruct
- [DeepSeekCoderv2](https://github.com/deepseek-ai/DeepSeek-Coder-V2?tab=readme-ov-file#2-model-downloads) 16b and 236b mixture of experts coding models with 128k context length
