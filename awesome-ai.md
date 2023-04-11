# Table of Contents

- [Large Language Models](#large-language-models)
  - [LLaMA models](#llama-models)
  - [Other SOTA Open Source Models](#other-sota-open-source-models)
  - [Data sets](#data-sets)
  - [Research](#research)
  - [LLM GUIs](#llm-guis)
    - [OpenAI](#openai)
    - [Other GUIs](#others)
  - [LLM Wrappers](#llm-wrappers)
  - [Showcases](#showcases)
  - [Fine Tuning](#fine-tuning)
- [Image Generation](#image-generation)
  - [Models](#models)
  - [Wrappers & GUIs](#wrappers--guis)
  - [Fine Tuning](#fine-tuning-1)
- [Benchmarking](#benchmarking)
- [Video](#video)
  - [Text to video generation](#text-to-video-generation)
  - [Frame Interpolation (Temporal Interpolation)](#frame-interpolation-temporal-interpolation)
  - [Super Resolution (Spacial Interpolation)](#super-resolution-spacial-interpolation)
  - [Spacio Temporal Interpolation](#spacio-temporal-interpolation)
- [Audio](#audio)
  - [Compression](#compression)
  - [Speech Recognition](#speech-recognition)
- [AI DevOps](#ai-devops)
- [Optimization](#optimization)
  - [Inference](#inference)
  - [Training](#training)
  - [Other Optimization](#other-optimization)

# Large Language Models



## LLaMA models
| Model                                                                                                                                                                                                         | Author            | Foundation | Size | Quantization | Fine Tuning Dataset                                                                                                                                                                                                                   | Format                    | LoRa |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|------------|------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|------|
| [vicuna-13b-GPTQ-4bit-128g](https://huggingface.co/4bit/vicuna-13b-GPTQ-4bit-128g)                                                                                                                            | 4bit              | vicuna     | 13b  | 4bit GPTQ    | Vicuna                                                                                                                                                                                                                                | safetensors               |      |
| [vicuna-13b](https://huggingface.co/helloollel/vicuna-13b)                                                                                                                                                    | helloollel        | vicuna     | 13b  | 16bit        | Vicuna                                                                                                                                                                                                                                | native .bin               |      |
| [vicuna-13b-GPTQ-4bit-128g](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g)                                                                                                                  | anon8231489123    | vicuna     | 13b  | 4bit GPTQ    | Vicuna                                                                                                                                                                                                                                | safetensors               |      |
| [vicuna-13b](https://huggingface.co/jeffwan/vicuna-13b/tree/main)                                                                                                                                             | jeffwan           | vicuna     | 13b  | ?            | Vicuna                                                                                                                                                                                                                                | ?                         |      |
| [ggml-vicuna-13b-4bit](https://huggingface.co/eachadea/ggml-vicuna-13b-4bit)                                                                                                                                  | eachadea          | vicuna     | 13b  | 4bit GPTQ?   | Vicuna                                                                                                                                                                                                                                | ggml .bin                 |      |
| [vicuna-13b](https://huggingface.co/eachadea/vicuna-13b)                                                                                                                                                      | eachadea          | vicuna     | 13b  | 16bit        | Vicuna                                                                                                                                                                                                                                | ?                         |      |
| [Vicuna-13B](https://huggingface.co/ShreyasBrill/Vicuna-13B)                                                                                                                                                  | SheryasBrill      | vicuna     | 13b  | 4bit (GPTQ?) | [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) [llama](https://huggingface.co/datasets/viewv/LLaMA-13B) [shareGPT unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)                  | ggml .bin / safetensors   |      |
| [vicuna-13b-8bit](https://huggingface.co/samwit/vicuna-13b-8bit)                                                                                                                                              | samwit            | vicuna     | 13b  | 8bit         |                                                                                                                                                                                                                                       | ?                         |      |
| [Vicuna-13b](https://huggingface.co/titan087/Vicuna-13b)                                                                                                                                                      | titan087          | vicuna     | 13b  | 16bit        |                                                                                                                                                                                                                                       | ?                         |      |
| [vicuna-13b-4bit](https://huggingface.co/elinas/vicuna-13b-4bit)                                                                                                                                              | elinas            | vicuna     | 13b  | 4bit GPTQ    |                                                                                                                                                                                                                                       | safetensors               |      |
| [vicuna-13b-delta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0)                                                                                                                                       | lmsys             | vicuna     | 13b  | 16bit        |                                                                                                                                                                                                                                       | ?                         |      |
| [vicuba-7b-int4-128g](https://huggingface.co/gozfarb/vicuba-7b-int4-128g) Quantized from [AlekseyKorshuk/vicuna-7b](https://huggingface.co/AlekseyKorshuk/vicuna-7b)                                          | gozfarb           | vicuna     | 7b   | 4bit GPTQ    | [ShareGPT Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)                                                                                                                                                                                     | safetensors               |      |
| [vicuna-AlekseyKorshuk-7B-GPTQ-4bit-128g](https://huggingface.co/TheBloke/vicuna-AlekseyKorshuk-7B-GPTQ-4bit-128g) quantized from [AlekseyKorshuk/vicuna-7b](https://huggingface.co/AlekseyKorshuk/vicuna-7b) | TheBloke          | vicuna     | 7b   | 4bit GPTQ    | [ShareGPT Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)                                                                                                                                                                                      | safetensors               |      |
| [Vicuna-7B-4bit-ggml](https://huggingface.co/Sosaka/Vicuna-7B-4bit-ggml)                                                                                                                                      | Sosaka            | vicuna     | 7b   | 4bit GPTQ?   | Vicuna                                                                                                                                                                                                                                | ggml                      |      |
| [vicuna-7b](https://huggingface.co/helloollel/vicuna-7b)                                                                                                                                                      | helloollel        | vicuna     | 7b   | 16bit        | Vicuna                                                                                                                                                                                                                                |                           |      |
| [ggml-vicuna-7b-4bit](https://huggingface.co/eachadea/ggml-vicuna-7b-4bit) based on AlekseyKorshuk/vicuna-7b                                                                                                  | eachadea          | vicuna     | 7b   | 4bit         | Vicuna                                                                                                                                                                                                                                | new ggml                  |      |
| [vicuna-7b](https://huggingface.co/AlekseyKorshuk/vicuna-7b)                                                                                                                                                  | AlekseyKorshuk    | vicuna     | 7b   | ?            | [ShareGPT Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)                                                                                                                                                                                      | native .bin               |      |
| [alpaca-30b-lora-int4](https://huggingface.co/elinas/alpaca-30b-lora-int4)                                                                                                                                    | elinas            | alpaca     | 30b  | 4bit GPTQ    | Alpaca                                                                                                                                                                                                                                | safetensors               |      |
| [Alpaca-30B-Int4-128G-Safetensors](https://huggingface.co/MetaIX/Alpaca-30B-Int4-128G-Safetensors)                                                                                                            | MetalX            | alpaca     | 30b  | 4bit         | Clean Alpaca dataset of 2023-04-06 using Chansung ALpaca Lora                                                                                                                                                                         | safetensor                |      |
| [alpaca-30B-ggml](https://huggingface.co/Pi3141/alpaca-30B-ggml)                                                                                                                                              | Pi3141            | alpaca     | 30b  | 4bit GPTQ    |                                                                                                                                                                                                                                       | ggml .bin                 |      |
| [alpaca-30b](https://huggingface.co/baseten/alpaca-30b)                                                                                                                                                       | baseten           | alpaca     | 30b  | 16bit        |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [gpt4-x-alpaca-13b-native-4bit-128g-cuda](https://huggingface.co/tsumeone/gpt4-x-alpaca-13b-native-4bit-128g-cuda)                                                                                            | tsumeone          | alpaca     | 13b  | 4bit         | [GPTeacher](https://github.com/teknium1/GPTeacher)                                                                                                                                                                                    | cuda safetensors          |      |
| [ggml-gpt4-x-alpaca-13b-native-4bit](https://huggingface.co/eachadea/ggml-gpt4-x-alpaca-13b-native-4bit)                                                                                                      | eachadea          | alpaca     | 13b  | 4bit         | Alpaca                                                                                                                                                                                                                                | ggml                      |      |
| [gpt4all-alpaca-oa-codealpaca-lora-13b](https://huggingface.co/jordiclive/gpt4all-alpaca-oa-codealpaca-lora-13b)                                                                                              | jordiclive        | alpaca     | 13b  | ?            | Nebulous/gpt4all_pruned, sahil2801/CodeAlpaca-20k, yahma/alpaca-cleaned, part of OpenAssistant                                                                                                                                        | native .bin               | yes  |
| [gpt4-x-alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)                                                                                                                                                | chavinlo          | alpaca     | 13b  | 16bit        | [GPTeacher](https://github.com/teknium1/GPTeacher)                                                                                                                                                                                    | native .bin               |      |
| [gpt4-x-alpaca-13b-native-4bit-128g](https://huggingface.co/Selyam/gpt4-x-alpaca-13b-native-4bit-128g)                                                                                                        | Selyam            | alpaca     | 13b  | 4bit GPTQ    | [GPTeacher](https://github.com/teknium1/GPTeacher)                                                                                                                                                                                    | triton / cuda             |      |
| [gpt4-x-alpaca-13b-native-4bit-128g](https://huggingface.co/anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g)                                                                                                | anon8231489123    | alpaca     | 13b  | 4bit GPTQ    | [GPTeacher](https://github.com/teknium1/GPTeacher)                                                                                                                                                                                    | triton / cuda             |      |
| [alpaca-13B-ggml](https://huggingface.co/Pi3141/alpaca-13B-ggml)                                                                                                                                              | Pi3141            | alpaca     | 13b  | 4bit GPTQ    |                                                                                                                                                                                                                                       | ggml .bin                 | yes  |
| [alpaca-30b-lora-int4](https://huggingface.co/elinas/alpaca-30b-lora-int4)                                                                                                                                    | elinas            | alpaca     | 13b  | 4bit GPTQ    |                                                                                                                                                                                                                                       | hf .pt                    | yes  |
| [alpaca-13b](https://huggingface.co/Dogge/alpaca-13b)                                                                                                                                                         | Dogge             | alpaca     | 13b  | 4bit GPTQ    |                                                                                                                                                                                                                                       | native .bin               |      |
| [alpaca-lora-13b](https://huggingface.co/baruga/alpaca-lora-13b)                                                                                                                                              | baruga            | alpaca     | 13b  | 8bit         |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b)                                                                                                                                            | chansung          | alpaca     | 13b  | 8bit         |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [llama-alpaca-stuff](https://huggingface.co/Draff/llama-alpaca-stuff/tree/main/Alpaca-Loras)                                                                                                                  | Draff             | alpaca     | 13b  | 8bit         |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [alpaca13B-lora](https://huggingface.co/samwit/alpaca13B-lora)                                                                                                                                                | samwit            | alpaca     | 13b  | 8bit         |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [gpt4all-alpaca-oa-codealpaca-lora-7b](https://huggingface.co/jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b)                                                                                                | jordiclive        | alpaca     | 7b   | ?            | [gpt4all_pruned](https://huggingface.co/datasets/Nebulous/gpt4all_pruned)<br />[CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)<br />[alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | ?                         | yes  |
| [alpaca-7b-nativeEnhanced](https://huggingface.co/8bit-coder/alpaca-7b-nativeEnhanced)                                                                                                                        | 8bit-coder        | alpaca     | 7b   | 16bit        |                                                                                                                                                                                                                                       | native .bin               |      |
| [alpaca-native-7B-4bit-ggjt](https://huggingface.co/LLukas22/alpaca-native-7B-4bit-ggjt) converted from Sosaka/Alpaca-native-4bit-ggml                                                                        | LLukas22          | alpaca     | 7b   | 4bit GPTQ?   |                                                                                                                                                                                                                                       | ggjt                      |      |
| [Alpaca-native-4bit-ggml](https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml)                                                                                                                              | Sosaka            | alpaca     | 7b   | 4bit GPTQ    |                                                                                                                                                                                                                                       | ggml .bin                 |      |
| [alpaca-7B-ggml](https://huggingface.co/Pi3141/alpaca-7B-ggml)                                                                                                                                                | Pi3141            | alpaca     | 7b   | 4bit GPTQ    |                                                                                                                                                                                                                                       | ggml .bin                 | yes  |
| [alpaca-native-4bit](https://huggingface.co/ozcur/alpaca-native-4bit)                                                                                                                                         | ozcur             | alpaca     | 7b   | 4bit GPTQ    |                                                                                                                                                                                                                                       | hf .pt                    |      |
| [alpaca-13b-lora-int4](https://huggingface.co/elinas/alpaca-13b-lora-int4)                                                                                                                                    | elinas            | alpaca     | 7b   | 4bit GPTQ    |                                                                                                                                                                                                                                       | safetensors               | yes  |
| [alpaca-native](https://huggingface.co/chavinlo/alpaca-native)                                                                                                                                                | chavinlo          | alpaca     | 7b   | 16bit        |                                                                                                                                                                                                                                       | native .bin               |      |
| [alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b/)                                                                                                                                                | tloen             | alpaca     | 7b   | 4bit GPTQ    |                                                                                                                                                                                                                                       | native .bin               | yes  |
| [llama-65B-ggml](https://huggingface.co/Pi3141/llama-65B-ggml)                                                                                                                                                | Pi3141            | llama      | 65b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | ggml .bin                 |      |
| [llama30b-lora-cot](https://huggingface.co/magicgh/llama30b-lora-cot)                                                                                                                                         | magicgh           | llama      | 30b  | ?            | Alpaca-CoT                                                                                                                                                                                                                            | .bin                      | yes  |
| [oasst-llama30b-ggml-q4](https://huggingface.co/Black-Engineer/oasst-llama30b-ggml-q4)                                                                                                                        | Black-Engineer    | llama      | 30b  | 4bit         | Open-Assistant, Alpaca                                                                                                                                                                                                                | ggml                      |      |
| [llama-30B-ggml](https://huggingface.co/Pi3141/llama-30B-ggml)                                                                                                                                                | Pi3141            | llama      | 30b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | ggml .bin                 |      |
| [instruct-13b-4bit-ggml](https://huggingface.co/llama-anon/instruct-13b-4bit-ggml)                                                                                                                            | llama-anon        | llama      | 13b  | 4bit GPTQ?   | instruct-13b weights                                                                                                                                                                                                                  | ggml .bin                 |      |
| [instruct-13b-4bit-128g](https://huggingface.co/gozfarb/instruct-13b-4bit-128g) from [llama-anon/instruct-13b](https://huggingface.co/llama-anon/instruct-13b)                                                | gozfarb           | llama      | 13b  | 4bit GPTQ?   | instruct-13b weights                                                                                                                                                                                                                  | safetensors               |      |
| [llama-13b-4bit-gr128](https://huggingface.co/4bit/llama-13b-4bit-gr128)                                                                                                                                      | 4bit              | llama      | 13b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | hf .pt                    |      |
| [llama13b-lora-gpt4all](https://huggingface.co/magicgh/llama13b-lora-gpt4all)                                                                                                                                 | magicgh           | llama      | 13b  | ?            | GPT4All, Alpaca-CoT                                                                                                                                                                                                                   | .bin                      | yes  |
| [llama-13b-pretrained-sft-do2](https://huggingface.co/dvruette/llama-13b-pretrained-sft-do2)                                                                                                                  | dvruette          | llama      | 13b  | 16bit        | ?                                                                                                                                                                                                                                     | native .bin               |      |
| [oasst-llama13b-ggml](https://huggingface.co/Black-Engineer/oasst-llama13b-ggml)                                                                                                                              | Black-Engineer    | llama      | 13b  | 16bit        | Open-Assistant                                                                                                                                                                                                                        | ? .bin                    |      |
| [koala-13B-HF](https://huggingface.co/TheBloke/koala-13B-HF)                                                                                                                                                  | TheBloke          | llama      | 13b  | 16bit        | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | native .bin               |      |
| [llama-13b-pretrained-sft-do2-ggml-q4](https://huggingface.co/Black-Engineer/llama-13b-pretrained-sft-do2-ggml-q4)                                                                                            | Black-Engineer    | llama      | 13b  | 4bit         | ?                                                                                                                                                                                                                                     | ggml                      |      |
| [oasst-llama13b-ggml-q4](https://huggingface.co/Black-Engineer/oasst-llama13b-ggml-q4)                                                                                                                        | Black-Engineer    | llama      | 13b  | 4bit         | Open-Assistant, Alpaca                                                                                                                                                                                                                | ggml                      |      |
| [oasst-llama13b-4bit-128g](https://huggingface.co/gozfarb/oasst-llama13b-4bit-128g)                                                                                                                           | gozfarb           | llama      | 13b  | 4bit         | Open-Assistant                                                                                                                                                                                                                        | safetensors               |      |
| [koala-13B-GPTQ-4bit-128g](https://huggingface.co/TheBloke/koala-13B-GPTQ-4bit-128g)                                                                                                                          | TheBloke          | llama      | 13b  | 4bit GPTQ    | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | triton / cuda safetensors |      |
| [koala-13B-GPTQ-4bit-128g-GGML](https://huggingface.co/TheBloke/koala-13B-GPTQ-4bit-128g-GGML)                                                                                                                | TheBloke          | llama      | 13b  | 4bit GPTQ    | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | ggml                      |      |
| [llama-30b-4bit](https://huggingface.co/kuleshov/llama-30b-4bit)                                                                                                                                              | kuleshov          | llama      | 13b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | hf .pt                    |      |
| [llama-30b-int4](https://huggingface.co/elinas/llama-30b-int4)                                                                                                                                                | elinas            | llama      | 13b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | hf .pt                    |      |
| [llama-30b-int4](https://huggingface.co/TianXxx/llama-30b-int4)                                                                                                                                               | TianXxx           | llama      | 13b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | hf .pt                    |      |
| [llama-13B-ggml](https://huggingface.co/Pi3141/llama-13B-ggml)                                                                                                                                                | Pi3141            | llama      | 13b  | 4bit GPTQ    | none                                                                                                                                                                                                                                  | ggml .bin                 |      |
| [koala-7b-ggml-unquantized](https://huggingface.co/TheBloke/koala-7b-ggml-unquantized)                                                                                                                        | TheBloke          | llama      | 7b   | 16bit        | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | ggml                      |      |
| [koala-7B-HF](https://huggingface.co/TheBloke/koala-7B-HF)                                                                                                                                                    | TheBloke          | llama      | 7b   | 16bit        | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | native .bin               |      |
| [koala-7B-GPTQ-4bit-128g](https://huggingface.co/TheBloke/koala-7B-GPTQ-4bit-128g)                                                                                                                            | TheBloke          | llama      | 7b   | 4bit GPTQ    | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | triton / cuda safetensors |      |
| [koala-7B-GPTQ-4bit-128g-GGML](https://huggingface.co/TheBloke/koala-7B-GPTQ-4bit-128g-GGML)                                                                                                                  | TheBloke          | llama      | 7b   | 4bit GPTQ    | Llama, ShareGPT, HC3 English, LAION OIC, Alpaca, ANthropic HH, WebGPT, Summarization                                                                                                                                                  | ggml                      |      |
| [gpt4all-lora](https://huggingface.co/nomic-ai/gpt4all-lora)                                                                                                                                                  | nomic-ai          | llama      | 7b   | ?            | [gpt4all prompt generations](https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations)                                                                                                                                     | ?                         | yes  |
| [llama-7B-ggml](https://huggingface.co/Pi3141/llama-7B-ggml)                                                                                                                                                  | Pi3141            | llama      | 7b   | 4bit GPTQ    | none                                                                                                                                                                                                                                  | ggml .bin                 |      |
| [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)                                                                                                                                           | decapoda-research | llama      | 7b   | 16bit        | none                                                                                                                                                                                                                                  | hf .pt                    |      |


## Other SOTA Open Source models
- [Cerebras GPT-13b](https://huggingface.co/cerebras) ([release notes](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/))
- [LAION OpenFlamingo | Multi Modal Model and training architecture](https://github.com/mlfoundations/open_flamingo)
- [TheBloke/galpaca-13b-gptq-4bit-128g](https://huggingface.co/TheBloke/galpaca-30B-GPTQ-4bit-128g), GALACTICA 30B fine tuned with Alpaca 
- [GeorgiaTechResearchInstitute/galpaca-6.7b](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-6.7b) GALACTICA 6.7B fine tuned with Alpaca

| Model                                                                                            | Author       | Foundation | Size | Quantization | Fine Tuning Dataset                                                                                                                                          | Format      | LoRa |
|--------------------------------------------------------------------------------------------------|--------------|------------|------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------|
| [oasst-pythia-12b-pretrained](https://huggingface.co/dvruette/oasst-pythia-12b-pretrained)       | dvruette     | Pythia     | 12b  | 16bit        | Open-Assistant?                                                                                                                                              | native .bin |      |
| [pythia-12b-pre-2000](https://huggingface.co/andreaskoepf/pythia-12b-pre-2000)                   | andreaskoepf | Pythia     | 12b  | 16bit        | joke, webgpt, gpt4all, alpaca, code_alpaca, minimath, codegen, testgen, grade school math, recipes, cmu wiki, oa wiki, prosocial dialogue, explain prosocial | native .bin |      |
| [pythia-12b-pre-3500](https://huggingface.co/andreaskoepf/pythia-12b-pre-3500)                   | andreaskoepf | Pythia     | 12b  | 16bit        | joke, webgpt, gpt4all, alpaca, code_alpaca, minimath, codegen, testgen, grade school math, recipes, cmu wiki, oa wiki, prosocial dialogue, explain prosocial | native .bin |      |
| [oasst-pythia-12b-reference](https://huggingface.co/dvruette/oasst-pythia-12b-reference)         | dvruette     | Pythia     | 12b  | 16bit        | ?                                                                                                                                                            | native .bin |      |
| [pythia-6.9b-gpt4all-pretrain](https://huggingface.co/andreaskoepf/pythia-6.9b-gpt4all-pretrain) | andreaskoepf | Pythia     | 6.9b | 16bit        | Open-Assistant? gpt4all?                                                                                                                                     | native .bin |

## Data sets
- [Alpaca-lora instruction finetuned using Low Rank Adaption](https://github.com/tloen/alpaca-lora)
- [codealpaca Instruction training data set for code generation](https://github.com/sahil280114/codealpaca)
- [LAION AI creates a cowd sourced fine-tuning data set for a future open source instruction based LLM](https://open-assistant.io) (https://github.com/LAION-AI/Open-Assistant / https://projects.laion.ai/Open-Assistant/)
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
- 

## Research
- [LLM Model Cards](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs)
- [GPTs are GPTs: An early look at the labor market impact potential of LLMs](https://arxiv.org/abs/2303.10130)
- [ViperGPT Visual Inference via Python Execution for reasoning](https://viper.cs.columbia.edu/)
- [Emergent Abilities of LLMs ](https://openreview.net/forum?id=yzkSU5zdwD), [blog post](https://www.jasonwei.net/blog/emergence)
- [visualchatgpt | Microsoft research proposes a multi-modal architecture to give chatgpt the ability to interpret and generate images based on open source foundation models](https://github.com/microsoft/visual-chatgpt)
- [facts checker reinforcement](https://arxiv.org/abs/2302.12813)

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
- [Auto GPT](https://github.com/Torantulino/Auto-GPT)
- [cheetah | Speech to text for remote coding interviews, giving you hints from GTP3/4](https://github.com/leetcode-mafia/cheetah)

### Other GUIs
- [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
- [Text Generation Webui | An all purpose UI to run LLMs of all sorts with optimizations](https://github.com/oobabooga/text-generation-webui) ([running LLaMA on less than 10GB vram](https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134), [running LLaMA-7b on a 3080](https://github.com/TimDettmers/bitsandbytes/issues/30#issuecomment-1455993902), [detailed guide](https://rentry.org/llama-tard-v2))
- [Alpaca-LoRa-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve)
- [ChatLLaMA | another implementation](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Dalai](https://github.com/cocktailpeanut/dalai)
- [ChatLLaMA | LLaMA-based ChatGPT for single GPUs](https://github.com/juncongmoo/chatllama)
- [Chatbot web app + HTTP and Websocket endpoints for BLOOM-176B inference with the Petals client](https://github.com/borzunov/chat.petals.ml)
- [Vicuna FastChat](https://github.com/lm-sys/FastChat)
- [Lit-llama](https://github.com/Lightning-AI/lit-llama)
- [gpt4all](https://github.com/nomic-ai/gpt4all)
- [openplayground Try out almost any LLM in a gui](https://github.com/nat/openplayground)


## LLM Wrappers
- [acheong08/ChatGPT Python](https://github.com/acheong08/ChatGPT)
- [mpoon/gpt-repository-loader](https://github.com/mpoon/gpt-repository-loader)
- [LangChain | framework for developing LLM applications](https://github.com/hwchase17/langchain) ([example](https://www.youtube.com/watch?v=iRJ4uab_NIg&t=588s))
- [LangFlow | GUI for Langchain](https://github.com/logspace-ai/langflow)
- [pyllama | hacked version of LLaMA based on Meta's implementation, optimized for Single GPUs](https://github.com/juncongmoo/pyllama)
- [Toolformer implementation | Allows LLMs to use Tools](https://github.com/lucidrains/toolformer-pytorch)
- [FastLLaMA Python wrapper for llama.cpp](https://github.com/PotatoSpudowski/fastLLaMa)
- [LlamaIndex (GPT Index) connecting LLMs to external data](https://github.com/jerryjliu/llama_index)

## Showcases
- [Opinionate.io AI Debating AI](https://opinionate.io/)
- [phind.com](phind.com) Developer Search Engine

## Fine Tuning
- [simple llama finetuner](https://github.com/lxe/simple-llama-finetuner)
- [LLaMA-LoRA Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner)
- [alpaca-lora](https://github.com/tloen/alpaca-lora)

# Image Generation

## Models
- https://github.com/kakaobrain/karlo
- https://lukashoel.github.io/text-to-room/
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

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
- [ermine-ai | Whisper in the browser using transformers.js](https://github.com/vishnumenon/ermine-ai)

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
- https://petals.ml/
- https://github.com/facebookincubator/AITemplate

## Training
- https://github.com/learning-at-home/hivemind

## Other Optimization
- https://github.com/HazyResearch/flash-attention
- https://github.com/stochasticai/x-stable-diffusion
- 

# Benchmarking
- https://videoprocessing.ai/benchmarks/
- https://paperswithcode.com/
- [Pythia | interpretability analysis for autoregressive transformers during training](https://github.com/EleutherAI/pythia)
