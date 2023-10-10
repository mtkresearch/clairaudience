# Zero-shot Domain-sensitive Speech Recognition with Prompt-conditioning Fine-tuning

Feng-Ting Liao, Yung-Chieh Chan, Yi-Chang Chen, Chan-Jan Hsu, Da-shan Shiu

[Paper Link](https://arxiv.org/abs/2307.10274)

*In this work, we propose a method to create domain-sensitive speech recognition models that utilize textual domain information by conditioning its generation on a given text prompt. This is accomplished by fine-tuning a pre-trained, end-to-end model (Whisper) to learn from demonstrations with prompt examples. We show that this ability can be generalized to different domains and even various prompt contexts, with our model gaining a Word Error Rate (WER) reduction of up to 33% on unseen datasets from various domains, such as medical conversation, air traffic control communication, and financial meetings. Considering the limited availability of audio-transcript pair data, we further extend our method to text-only fine-tuning to achieve domain sensitivity as well as domain adaptation. We demonstrate that our text-only fine-tuned model can also attend to various prompt contexts, with the model reaching the most WER reduction of 29% on the medical conversation dataset.*

## Installation
Please first install openai's whisper repo and also the packages in `requirements.txt`

## Training
To run the training example, ensure that Gigaspeech medium is downladed to `data/hf_dd_data/gigaspeech/m` and execute
```
python ./clairaudience/main.py ./configs/cfg_gigaspeech_ft_base.json
```

## Evaluation
To run the training example, ensure that Gigaspeech medium is downladed to `data/hf_dd_data/gigaspeech/m` and execute
```
python ./clairaudience/main.py ./configs/cfg_gigaspeech_evaluation.json
```

## Model Weight
See https://huggingface.co/MediaTek-Research/Clairaudience
