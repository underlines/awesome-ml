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
