[🏠Home](README.md)

# Audio

## Compression

- [EnCodec](https://github.com/facebookresearch/encodec) SOTA deep learning based audio codec supporting both mono 24 kHz audio and stereo 48 kHz audio

## Multiple Tasks

- [audio-webui](https://github.com/gitmylo/audio-webui) A web-based UI for various audio-related Neural Networks with features like text-to-audio, voice cloning, and automatic-speech-recognition using Bark, AudioLDM, AudioCraft, RVC, coqui-ai and Whisper
- [tts-generation-webui](https://github.com/rsxdalv/tts-generation-webui) for all things TTS, currently supports Bark v2, MusicGen, Tortoise, Vocos
- [Speechbrain](https://github.com/speechbrain/speechbrain) A PyTorch-based Speech Toolkit for TTS, STT, etc
- [Nvidia NeMo](https://github.com/NVIDIA/NeMo) TTS, LLM, Audio Synthesis framework
- [speech-rest-api](https://github.com/askrella/speech-rest-api) for Speech-To-Text and Text-To-Speech with Whisper and Speechbrain
- [LangHelper](https://github.com/NsLearning/LangHelper) language learning through Text-to-speech + chatGPT + speech-to-text to practise speaking assessments, memorizing words and listening tests
- [Silero-models](https://github.com/snakers4/silero-models) pre-trained speech-to-text, text-to-speech and text-enhancement for ONNX, PyTorch, TensorFlow, SSML
- [AI-Waifu-Vtuber](https://github.com/ardha27/AI-Waifu-Vtuber) AI Waifu Vtuber & is a virtual streamer. Supports multiple languages and uses VoiceVox, DeepL, Whisper, Seliro TTS, and VtubeStudio, and now also supports Twitch streaming.
- [Voicebox](https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/) large-scale text-guided generative speech model using non-autoregressive flow-matching, [paper](https://research.facebook.com/publications/voicebox-text-guided-multilingual-universal-speech-generation-at-scale/), [demo](https://voicebox.metademolab.com), [pytorch implementation](https://github.com/jaloo555/voicebox-pytorch), [implementation](https://github.com/SpeechifyInc/Meta-voicebox)
- [Auto-Synced-Translated-Dubs](https://github.com/ThioJoe/Auto-Synced-Translated-Dubs) Automatic YouTube video speech to text, translation, text to speech in order to dub a whole video
- [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) Foundational Models for SOTA Speech and Text Translation
- [Amphion](https://github.com/open-mmlab/Amphion) a toolkit for Audio, Music, and Speech Generation supporting TTS, SVS, VC, SVC, TTA, TTM
- [voicefixer](https://github.com/haoheliu/voicefixer) restore human speech regardless how serious its degraded
- [VoiceCraft](https://github.com/jasonppy/VoiceCraft) clone and edit an unseen voice with few seconds example and Text-to-Speech capabilities
- [audapolis](https://github.com/bugbakery/audapolis) an audio/video editor for spoken word media editing like a text editor using speech recognition
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) is a multi lingual voice generation model that supports inference, training, and deployment, zero-shot, cross-lingual voice cloning, and instruction-following capabilities, with features like Flow matching training, Repetition Aware Sampling inference, and streaming inference mode
- [Speech-AI-Forge](https://github.com/lenML/Speech-AI-Forge) is a gradio GUI and API server supporting multiple tasks and models such as ChatTTS, FishSpeech, CosyVoice, FireRedTTS for TTS, Whisper for ASR, and OpenVoice for voice conversion, with functionalities like speaker switching, style controls, long text inference, SSML scripting, and voice creation

## Speech Recognition

- [Whisper](https://github.com/openai/whisper) SOTA local open-source speech recognition in many languages and translation into English
  - [Whisper JAX implementation](https://github.com/sanchit-gandhi/whisper-jax) runs around 70x faster on CPU, GPU and TPU
  - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) C/C++ port for Intel and ARM based Mac OS, ANdroid, iOS, Linux, WebAssembly, Windows, Raspberry Pi
  - [faster-whisper-livestream-translator](https://github.com/JonathanFly/faster-whisper-livestream-translator) A buggy proof of concept for real-time translation of livestreams using Whisper models, with suggestions for improvements including noise reduction and dual language subtitles
  - [Buzz](https://github.com/chidiwilliams/buzz) Mac GUI for Whisper
  - [whisperX](https://github.com/m-bain/whisperX) Fast automatic speech recognition (70x realtime with large-v2) using OpenAI's Whisper, word-level timestamps, speaker diarization, and voice activity detection
  - [distil-whisper](https://github.com/huggingface/distil-whisper) a distilled version of Whisper that is 6 times faster, 49% smaller, and performs within 1% word error rate (WER) on out-of-distribution evaluation sets, [paper](https://arxiv.org/abs/2311.00430), [model](https://huggingface.co/distil-whisper/distil-large-v2), [hf](https://news.ycombinator.com/item?id=38093353)
  - [whisper-turbo](https://github.com/FL33TW00D/whisper-turbo) a fast, cross-platform Whisper implementation, designed to run entirely client-side in your browser/electron app.
  - [faster-whisper](https://github.com/systran/faster-whisper) Faster Whisper transcription with CTranslate2
  - [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) a command line client based on faster-whisper and compatible with the original client from openai/whisper
  - [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) a speaker diarization tool that is based on faster-whisper and NVIDIA NeMo
  - [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) portable ready to run binaries of faster-whisper for Windows
  - [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline) scalable, modular, end to end multi-speaker speech to text solution implemented using AzureML pipelines
  - [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) opinionated CLI to transcribe audio to text using whisper v3 on edge devices using optimum and flash attention
  - [insanely-fast-whisper-cli](https://github.com/ochen1/insanely-fast-whisper-cli) The fastest Whisper optimization for automatic speech recognition as a command-line interface
  - [WhisperLive](https://github.com/collabora/WhisperLive) real time transcription using voice activity detection and TensorRT or FasterWhisper backends
  - [Whisper Medusa](https://github.com/aiola-lab/whisper-medusa) speed improvements by multi token prediction per iteration maintaining almost similar quality
- [ermine-ai | Whisper in the browser using transformers.js](https://github.com/vishnumenon/ermine-ai)
- [wav2vec2 dimensional emotion model](https://github.com/audeering/w2v2-how-to)
- [MeetingSummarizer](https://github.com/rajpdus/MeetingSummarizer) using Whisper and GPT3.dd
- [Facebook MMS: Speech recognition of over 1000 languages](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [Moonshine](https://github.com/usefulsensors/moonshine) Speech to text models optimized for fast and accurate inference on edge devices outperforming Whisper
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) is a low-latency real time speech-to-text library, with advanced voice activity detection, wake word activation, and instant transcription using a combination of WebRTCVAD, SileroVAD, Faster_Whisper, and Porcupine or OpenWakeWord

voice activity detection (VAD):

- [Silero-VAD](https://github.com/snakers4/silero-vad) pre-trained enterprise-grade real tie Voice Activity Detector
- [libfvad](https://github.com/dpirch/libfvad) fork of WebRTC VAD engine as a standalone library independent from other WebRTC features
- [voice_activity_detection](https://github.com/filippogiruzzi/voice_activity_detection) Voice Activity Detection based on Deep Learning & TensorFlow
- [rVADfast](https://github.com/zhenghuatan/rVADfast) unsupervised, robust voice activity detection

subtitle generation:

- [subtitler](https://github.com/dmtrKovalenko/subtitler) on-device web app for audio transcribing and rendering subtitles
- [pyvideotrans](https://github.com/jianchang512/pyvideotrans) is a video translation and voiceover tool supporting STT, translation, TTS synthesis and audio separation, capable of translating videos into multiple languages while retaining background audio, and offering functionalities such as subtitle creation, batch translation, and audio-video merging

## TextToSpeech

- [Bark](https://github.com/suno-ai/bark) transformer-based text-to-audio model by Suno. Can generate highly realistic, multilingual speech and other audio like music, background noise and simple effects
  - [Bark-Voice-Clones](https://github.com/nikaskeba/Bark-Voice-Clones)
  - [Bark WebUI colab notebooks](https://github.com/camenduru/bark-colab)
  - [bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone)
  - [Bark Infinity for longer audio](https://github.com/JonathanFly/bark)
  - [Bark WebUI](https://github.com/makawy7/bark-webui)
  - [bark-voice-cloning-HuBERT-quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer) Voice cloning with bark in high quality using Python 3.10 and  ggingface models.
  - [bark-gui](https://github.com/C0untFloyd/bark-gui) Gradio Web UI for an extended Bark version, with long generation, cloning, SSML, for Windows and other platforms, supporting NVIDIA/Apple GPU/CPU, 
  - [bark-voice-cloning](https://github.com/KevinWang676/Bark-Voice-Cloning) for chinese speech, based on bark-gui by C0untFloyd
  - [Barkify](https://github.com/anyvoiceai/Barkify) unoffical training implementation of Bark TTS by suno-ai
  - [Bark-RVC](https://github.com/ORI-Muchim/BARK-RVC) Multilingual Speech Synthesis Voice Conversion using Bark + RVC
- [Coqui TTS | deep learning toolkit for Text-to-Speech](https://github.com/coqui-ai/TTS)
  - [Tutorial](https://www.youtube.com/watch?v=dfmlyXHQOwE) for Coqui VITS and Whisper to automate voice cloning and [Colab notebook](https://colab.research.google.com/drive/1Swo0GH_PjjAMqYYV6He9uFaq5TQsJ7ZH?usp=sharing#scrollTo=nSrZbKCXxalg)
- [StyleTTS2 implementation](https://github.com/yl4579/StyleTTS2) Elevenlabs quality TTS through Style Diffusion and Adversarial Training with Large Speech Language Models
- [StyleTTS implementation](https://github.com/yl4579/StyleTTS)
  - [StyleTTS-VC](https://github.com/yl4579/StyleTTS-VC) One-Shot Voice Conversion by Knowledge Transfer from Style-Based TTS Models
- [Vall-E and Vall-E X](https://valle-demo.github.io/), [paper](https://arxiv.org/abs/2301.02111), [code](https://github.com/enhuiz/vall-e). Zero Shot TTS preserving emotion, expression, similarity and allows language transfer
  - [Vall-e PyTorch Implementation](https://github.com/enhuiz/vall-e) of Vall-E based on EnCodec tokenizer
  - [Vall-E PyTorch implementation](https://github.com/lifeiteng/vall-e)
  - [Vall-E X](https://github.com/Plachtaa/VALL-E-X) open source implementation of Microsoft's VALL-E X zero-shot TTS model
- [NaturalSpeech implmenetation](https://github.com/heatz123/naturalspeech)
  - [naturalspeech2-pytorch](https://github.com/lucidrains/naturalspeech2-pytorch) Implementation of Natural Speech 2, Zero-shot Speech and Singing Synthesizer, in Pytorch
  - [NS2VC](https://github.com/adelacvg/NS2VC) Unofficial implementation of NaturalSpeech2 for Voice Conversion and Text to Speech
- [IMS Toucan, TTS Toolkit from University of Stuttgart](https://github.com/digitalphonetics/ims-toucan)
- [YourTTS | Zero Shot Multi Speaker TTS and Voice Conversion for everyone](https://github.com/Edresson/YourTTS)
- [PaddleSpeech | Easy to use Speech Toolkit with Self Supervised learning, SOTA Streaming with punctuation, TTS, Translation etc](https://github.com/PaddlePaddle/PaddleSpeech)
- [Tortoise TTS | Open source multi voice TTS system](https://github.com/neonbjb/tortoise-tts)
  - [finetune guide using DLAS DL-Art-School](https://www.youtube.com/watch?v=lnIq4SFFXWs), [Master Deep Voice Cloning in Minutes](https://youtu.be/OiMRlqcgDL0)
  - [DL-Art-School](https://github.com/152334H/DL-Art-School) fine tuning tortoise with DLAS GUI
  - [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) fast Tortoise TTS inference up to 5x. [Video tutorial](https://www.youtube.com/watch?v=8i4T5v1Fl_M)
  - [Tortoise mrq fork for voice cloning](https://git.ecker.tech/mrq/ai-voice-cloning)
- [piper](https://github.com/rhasspy/piper) A fast, local neural text to speech system that sounds great and is optimized for the Raspberry Pi 4. Using VITS and onnxruntime
- [PITS](https://github.com/anonymous-pits/pits) PyTorch implementation of Variational Pitch Inference for End-to-end Pitch-controllable TTS. [hf demo](https://huggingface.co/spaces/anonymous-pits/pits), [samples](https://anonymous-pits.github.io/pits/)
- [VoiceCloning](https://github.com/MartinMashalov/VoiceCloning) Implementing the [YourTTS paper](https://arxiv.org/abs/2112.02418) for Zero-Shot multi-speaker Attention-Based TTS using VITS approaches
  - [VITS-Umamusume-voice-synthesizer](https://huggingface.co/spaces/1raliopunche/VITS-Umamusume-voice-synthesizer) (Multilingual Anime TTS) Including Japanese TTS, Chinese and English TTS, speakers are all anime characters.
- [Parallel WaveGAN implementation in PyTorch](https://github.com/kan-bayashi/ParallelWaveGAN) for high quality text to speech synthesis [paper](https://github.com/kan-bayashi/ParallelWaveGAN)
- [real-time-voice](https://github.com/michaelcrubin/real-time-voice) SV2TTS voice cloning TTS implementation using WaveRNN, Tacatron, GE2E
- [voicebox-pytorch](https://github.com/lucidrains/voicebox-pytorch) Implementation of Voicebox, new SOTA Text-to-speech network from MetaAI, in Pytorch
- [MockingBird](https://github.com/babysor/MockingBird) Clone a voice in 5 seconds to generate arbitrary speech in real-time
- [XTTS-2-UI](https://github.com/BoltzmannEntropy/xtts2-ui) XTTS-2 Text-Based Voice Cloning
  - [xtts-webui](https://github.com/daswer123/xtts-webui) Webui for using XTTS
  - [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui) Slightly improved official version for finetune xtts
  - [XTTS-RVC-UI](https://github.com/Vali-98/XTTS-RVC-UI) A Gradio UI for XTTSv2 and RVC
- [EmotiVoice](https://github.com/netease-youdao/EmotiVoice) Multi-Voice and Prompt-Controlled TTS Engine
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech) inverted whisper (aka spear-tts) with voice cloning, language mixing using EnCodec and Vocos
- [metavoice](https://github.com/metavoiceio/metavoice-src) supporting emotional speech rhythm and tone, cloning for British and American English and cross lingual cloning with finetuning
- [StyleTTS2](https://github.com/yl4579/StyleTTS2) Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
  - [StyleTTS2 pip](https://github.com/sidharthrajaram/StyleTTS2) Python pip package for StyleTTS2
- [fish-speech](https://github.com/fishaudio/fish-speech) expressive TTS with generated voices with fine tuning (voice cloning) capabilities
- [ChatTTS](https://github.com/2noise/ChatTTS) optimized for dialogue-based TTS with natural andexpressive speech synthesis in English and Chinese with fine grained prosodic features like laughter, pauses and interjections
- [Parler-TTS](https://github.com/huggingface/parler-tts) Huggingface's Parler TTS model can generate high quality natural speech by using two prompts one for the text and one for the style supporting gender, pitch, speaking style etc, with a Mini 0.8B and 2.3B Large model released and ready for fine tuning supporting SDPA and Flash Attention 2
- [MeloTTS](https://github.com/myshell-ai/MeloTTS/tree/main) multi lingual multi accent TTS for English, Spanish, French, Chinese, Korean and Japanese even working on CPU inference and support for training new languages
- [F5-TTS](https://github.com/SWivid/F5-TTS) is a text-to-speech system that utilizes Diffusion Transformer with ConvNeXt V2 for fast training and inference, and implements Sway Sampling strategy to significantly enhance performance

## Voice Conversion

- [voicepaw/so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) SoftVC VITS Singing Voice Conversion Fork with realtime support and greatly improved interface. Based on so-vits-svc 4.0 (v1)
  - [Video tutorial by Nerdy Rodent](https://www.youtube.com/watch?v=tZn0lcGO5OQ)
  - [nateraw/so-vits-svc-fork gradio app](https://github.com/nateraw/voice-cloning) for inference of so-vits-svc-fork voice models + ([training in colab](https://colab.research.google.com/github/nateraw/voice-cloning/blob/main/training_so_vits_svc_fork.ipynb) with yt downloader and audio splitter, [hf space demo](https://hf.co/spaces/nateraw/voice-cloning))
  - [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
  - [LoRa svc](https://github.com/PlayVoice/lora-svc) singing voice conversion based on whisper, and lora
  - [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Voice Conversion and Text-to-Speech WebUI with instant cloning, fine tuning and cross lingual support
  - [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) simple and easy-to-use voice transformation (voice changer) web GUI based on VITS
    - [rvc-webui](https://github.com/ddPn08/rvc-webui) Win/Mac/Linux installer and Guide for RVC-Project
    - [RVC-GUI](https://github.com/Tiger14n/RVC-GUI) fork of RVC for easy audio file voice conversion locally, only inference, no training
    - [RVC Studio](https://github.com/SayanoAI/RVC-Studio) training RVC models and generating AI voice covers
    - [AI-Song-Cover-RVC](https://github.com/ardha27/AI-Song-Cover-RVC) All in One: Youtube WAV Download, Separating Vocals, Splitting Audio, Training, and Inference
    - [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio files
  - [w-okada/voice-changer](https://github.com/w-okada/voice-changer) supports MMVC, so-vits-svc, RVC, DDSP-SVC, processing offloading over LAN, real time conversion
  - [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) Real-time singing voice conversion based on DDSP, training and inference uses lower requirements than diff-svc and so-vits-svc
  - [Leader board of SOTA models](https://github.com/Anjok07/ultimatevocalremovergui/issues/344) for stem separation using model ensembles in UVR
  - [VITS GUI to load VITS text to speech models](https://github.com/CjangCjengh/MoeGoe_GUI)
  - [Vits-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning) pipeline of VITS finetuning for fast speaker adaptation TTS, and many-to-many voice conversion
  - [AI-Cover-Song](https://github.com/reshalfahsi/AI-Cover-Song) a google colab to do singing voice conversion with so-vits-svc-fork
  - [hf-rvc](https://github.com/esnya/hf-rvc) a package for RVC implementation using HuggingFace's transformers with the capability to convert from original unsafe models to HF models and voice conversion tasks
  - [VitsServer](https://github.com/LlmKira/VitsServer) A VITS ONNX server designed for fast inference
  - [jax-so-vits-svc-5.0](https://github.com/flyingblackshark/jax-so-vits-svc-5.0) Rewrite so-vits-svc-5.0 in jax
- [w-okada/voice-changer | real time voice conversion using various models like MMVC, so-vits-svc, RVC, DDSP-SVC](https://github.com/w-okada/voice-changer/blob/master/README_en.md)
- [Diff-svc](https://github.com/prophesier/diff-svc) Singing Voice Conversion via Diffusion model
  - [FastDiff implementation| Fast Conditional Diffusion Model for High-Quality Speech Synthesis](https://github.com/Rongjiehuang/FastDiff)
  - [Fish Diffusion](https://github.com/fishaudio/fish-diffusion) easy to understand TTS / SVS / SVC framework, can convert Diff models
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) abandoned project
  - [Real-Time-Voice-Cloning v2](https://github.com/liuhaozhe6788/voice-cloning-collab) active fork of the original for google collab
- [Raven with voice cloning 2.0](https://huggingface.co/spaces/Kevin676/Raven-with-Voice-Cloning-2.0/tree/main) by Kevin676
- [CoMoSpeech](https://github.com/zhenye234/CoMoSpeech) consistency model distilled from a diffusion-based teacher model, enabling high-quality one-step speech and singing voice synthesis
  - [CoMoSVC](https://github.com/Grace9994/CoMoSVC) One-Step Consistency Model Based Singing Voice Conversion & Singing Voice Clone
- [NS2VC](https://github.com/adelacvg/NS2VC) WIP Unofficial implementation of NaturalSpeech2 for Voice Conversion
- [vc-lm](https://github.com/nilboy/vc-lm) train an any-to-one voice conversion models, referncing vall-e, using encodec to create tokens and building a transformer language model on tokens
- [knn-vc](https://github.com/bshall/knn-vc) official implementation of Voice Conversion With Just Nearest Neighbors (kNN-VC) contains training and inference for any-to-any voice conversion model, [paper](https://arxiv.org/abs/2305.18975), [examples](https://bshall.github.io/knn-vc/)
- [FreeVC](https://github.com/OlaWod/FreeVC) High-Quality Text-Free One-Shot Voice Conversion including pretrained models [HF demo](https://huggingface.co/spaces/OlaWod/FreeVC), [examples](https://olawod.github.io/FreeVC-demo)
- [TriAAN-VC](https://github.com/winddori2002/TriAAN-VC) a Pytorch deep learning model for any-to-any voice conversion, with SOTA performance achieved by using an attention-based adaptive normalization block to extract target speaker representations while minimizing the loss of the source content. [demo](https://winddori2002.github.io/vc-demo.github.io/), [paper](https://arxiv.org/abs/2303.09057)
- [EasyVC](https://github.com/MingjieChen/EasyVC) toolkit supporting various encoders and decoders, focusing on challenging VC scenarios such as one-shot, emotional, singing, and real-time. [demo](https://mingjiechen.github.io/easyvc/index.html)
- [MoeVoiceStudio](https://github.com/NaruseMioShirakana/MoeVoiceStudio/tree/MoeVoiceStudio) GUI supporting JOKE, SoVits, DiffSvc, DiffSinger, RVC, FishDiffusion
- [OpenVoice](https://github.com/myshell-ai/openvoice) V2 with better quality and multilingual support and MIT licencing, Instant Voice Cloning with accuracte tone color, flexibel style control and zero-shot cross lingual voice cloning
- [NeuCoSVC](https://github.com/thuhcsi/NeuCoSVC?tab=readme-ov-file) official implementation of Neural Concatenative Singing Voice Conversion [paper](https://arxiv.org/abs/2312.04919)
- [talking-avatar-and-voice-cloning](https://github.com/alifallaha1/talking-avatar-and-voice-cloning) streamlit app with XTTS and SadTalker implementation featuring lip sync and avatar animation
- [AiVoiceClonerPRO](https://github.com/AryanVBW/AiVoiceClonerPRO) easy web interface using VITS and InterSpeech2023-RMVPE pitch extraction method

## Video Voice Dubbing

- [weeablind](https://github.com/FlorianEagox/weeablind) dub multi lingual media using modern AI speech synthesis, diarization, and language identification
- [Auto-synced-translated-dubs](https://github.com/ThioJoe/Auto-Synced-Translated-Dubs) Youtube audio translation and dubbing pipeline using Whisper speech-to-text, Google/DeepL Translate, Azure/Google TTS
- [videodubber](https://github.com/am-sokolov/videodubber) dub video using GCP TTS, Translate, Whisper, Spacy tokenization and syllable counting
- [TranslatorYouTuber](https://github.com/AdiKsOnDev/TranslatorYouTuber) Takes a youtube video, clones the voice and re-creates that video in a different language
- [global-video-dubbing](https://github.com/ZackAkil/global-video-dubbing) Using Googel Cloud Video Intelligence API with Cloud Translation API and Cloud Text to Speech API to generate voice dubbing and tranaslations in many languages automatically
- [wav2lip](https://github.com/Rudrabha/Wav2Lip) Lip Syncing from audio
  - [Wav2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN) High quality Lip sync with wav2lip + Tencent GFPGAN
- [video-retalking](https://github.com/OpenTalker/video-retalking) Audio-based Lip Synchronization for Talking Head Video Editing In the Wild
- [Wunjo AI](https://github.com/wladradchenko/wunjo.wladradchenko.ru) Synthesize & clone voices in English, Russian & Chinese, real-time speech recognition, deepfake face & lips animation, face swap with one photo, change video by text prompts, segmentation, and retouching. Open-source, local & free
- [YouTranslate](https://github.com/AdiKsOnDev/YouTranslate) Takes a youtube video, clones the voice with elevenlabs API translate the text with google translate API and re-creates that video in a different language
- [audio2photoreal](https://github.com/facebookresearch/audio2photoreal) Photoreal Embodiment by Synthesizing Humans including pose, hands and face in Conversations
- [TurnVoice](https://github.com/KoljaB/TurnVoice) Dubbing via CoquiTTS, Elevenlaps, OpenAI or Azure Voices, Translation, Speaking Style changer, Precise control via Editor, Background Audio Preservation
- [pyvideotrans](https://github.com/jianchang512/pyvideotrans) is a video translation and voiceover tool supporting STT, translation, TTS synthesis and audio separation, capable of translating videos into multiple languages while retaining background audio, and offering functionalities such as subtitle creation, batch translation, and audio-video merging
- [SoniTranslate](https://github.com/R3gm/SoniTranslate) is a gradio based GUI for video translation and dubbing, OpenAI API for transcription, translation, and TTS, and supporting various output formats and multi-speaker TTS, with features like vocal enhancement, voice imitation, and extensive language support
- [VideoLingo](https://github.com/Huanshere/VideoLingo) subtitle transcription and audio dubbing using WHisperX, LLMs, GPT-SoVITS

## Music Generation

- [audiocraft](https://github.com/facebookresearch/audiocraft) library for audio processing and generation with deep learning using EnCodec compressor / tokenizer and MusicGen support
  - audiocraft-infinity-webui webui supporting generation longer than 30 seconds, song continuation, seed option, load local models from chavinlo's [training repo](https://github.com/chavinlo/musicgen_trainer), MacOS/linux support, running on CPU/gpu
  - [musicgen_trainer](https://github.com/chavinlo/musicgen_trainer) simple trainer for musicgen/audiocraft
  - [audiocraft-webui](https://github.com/CoffeeVampir3/audiocraft-webui) basic webui with support for long audio, segmented audio and processing queue
  - [audiocraft-webui](https://github.com/sdbds/audiocraft-webui) another basic webui, unknown feature set
  - [MusicGeneration](https://github.com/vluz/MusicGeneration) a streamlit gui for audiocraft and musicgen
  - [audiocraftgui](https://github.com/DragonForgedTheArtist/audiocraftgui) with wxPython supporting continuous generation by using chunks and overlaps
  - [MusicGen](https://huggingface.co/spaces/facebook/MusicGen) a simple and controllable model for music generation using a Transformer model [examples](https://ai.honu.io/papers/musicgen/), [colab](https://colab.research.google.com/drive/1-Xe9NCdIs2sCUbiSmwHXozK6AAhMm7_i?usp=sharing), [colab collection](https://github.com/camenduru/MusicGen-colab)
  - [audiocraft-infinity-webui](https://github.com/1aienthusiast/audiocraft-infinity-webui) generation length  over 30 seconds, ability to continue songs, seeds, allows to load local models
  - [AudioCraft Plus](https://github.com/GrandaddyShmax/audiocraft_plus) an all-in-one WebUI for the original AudioCraft, adding multiband diffusion, continuation, custom model support, mono to stereo and more
- [AudioLDM](https://audioldm.github.io/) Generate speech, sound effects, music and beyond, with text [code](https://github.com/haoheliu/AudioLDM), [paper](https://arxiv.org/abs/2301.12503), [HF demo](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)
- [StableAudio](https://github.com/Stability-AI/stable-audio-tools) Stability AI's Stable Audio only providing Training and Inference code, no models
- [SoundStorm-Pytorch](https://github.com/lucidrains/soundstorm-pytorch) a Pytorch implementation of Google Deepmind's SoundStorm, applying MaskGiT to residual vector quantized codes from Soundstream, using a Conformer transformer architecture for efficient parallel audio generation from text instructions

## Audio Source Separation

- [Separate Anything You Describe](https://github.com/audio-agi/audiosep) Describe what you want to isolate from audio, Language-queried audio source separation (LASS), [paper](https://arxiv.org/abs/2308.05037v1)
- [Hybrid-Net](https://github.com/DoMusic/Hybrid-Net) Real-time audio source separation, generate lyrics, chords, beat by lamucal.ai
- [TubeSplitter](https://github.com/WNSTN92/TubeSplitter) Web application to extract and separate audio stems from YouTube videos using Flask, pytube, and spleeter
- [demucs](https://github.com/adefossez/demucs) Hybrid Transformer based source separation
  - [streamstem](https://github.com/otonomee/streamstem) web app utilizing yt-dlp, spotify-api and demucs for an end to end audio source separation pipeline
  - [moseca](https://github.com/fabiogra/moseca) web app for Music Source Separation & Karaoke utilizig demucs
  - [MISST](https://github.com/Frikallo/MISST) native windows GUI for demucs supporting youtube, spotify and files

## Research

- [Vocos](https://charactr-platform.github.io/vocos/) Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis
- [WavJourney](https://arxiv.org/abs/2307.14335) Compositional Audio Creation with LLMs [github](https://github.com/audio-agi/wavjourney)
- [PromptingWhisper](https://github.com/jasonppy/PromptingWhisper) Audio-Visual Speech Recognition, Code-Switched Speech Recognition, and Zero-Shot Speech Translation for Whisper
- [Translatotron 3](https://blog.research.google/2023/12/unsupervised-speech-to-speech.html) Unsupervised speech-to-speech translation from monolingual data

### Benchmarks

- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)
- [ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)