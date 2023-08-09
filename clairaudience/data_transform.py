import os
import sys
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Union
from whisper.normalizers import EnglishTextNormalizer
from whisper.audio import N_MELS, N_FRAMES
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from clairaudience.data_process import preprocess_gigaspeech_text
from clairaudience.model import whisper_feature_extractor
_CE_IGNORE_INDEX = -100
PREV_TOKEN_ID=50361    #"<|startofprev|>"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    r"""
    This is used with Dataloader api to get the kl_target
    """
    processor: Any
    generation_mode: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        # Train with decoder only
        if 'input_features' in features[0]:
            batch["input_features"] = torch.FloatTensor(np.array([feat["input_features"] for feat in features]))

        # Decoder input ids
        decoder_input_feats = [{"input_ids": feature["decoder_input_ids"]} for feature in features]
        if self.generation_mode:
            # pad to the left in generate mode
            self.processor.tokenizer.padding_side = 'left'
            padded_decoder_input = self.processor.tokenizer.pad(decoder_input_feats, return_tensors="pt")
            batch["decoder_input_ids"] = padded_decoder_input["input_ids"]
            batch["decoder_attention_mask"] = padded_decoder_input["attention_mask"]
            self.processor.tokenizer.padding_side = 'right'
        else:
            # pad to the right for training mode
            batch["decoder_input_ids"] = self.processor.tokenizer.pad(decoder_input_feats, return_tensors="pt")["input_ids"]
        
        if self.generation_mode:
            # collect prompt length to chop off prompt when generating
            batch["decoder_prompt_len"] = np.array([feat["decoder_prompt_len"] for feat in features])
            batch["text"] = np.array([feature["text"] for feature in features])
        else:
            # Labels for the model to regress on
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), _CE_IGNORE_INDEX)
            batch["labels"] = labels

        logging.debug({k:v.shape for k, v in batch.items()})
        return batch


def prompt_format_func(domains):
    """
    format list of domain into prompt
    """
    if len(domains) == 0:
        return ""
    return ("[ domain: " + ", ".join(domains) + "]")


def _load_pinfo(pinfo_str: Union[List[str], str])->Union[List[str], Dict[str, List[str]]]:
    if isinstance(pinfo_str, str):
        pinfo = pd.read_csv(pinfo_str)
        return pd.Series(pinfo.domain.values, index=pinfo.audio_id).apply(eval).to_dict()
    return pinfo_str


def _get_prompts(samples, 
                 pinfo: Union[List, Dict[str, List[str]]] = None, 
                 use_random_selection=True):
    batch_size = len(samples["text"])
    prompt_strs = []
    if isinstance(pinfo, list) and len(pinfo):
        prompt_strs = [np.random.choice(pinfo, 1)[0] for _ in range(batch_size)]
    elif (isinstance(pinfo, dict) and len(pinfo)) or "domain" in samples:
        # if we have domains
        for i in range(batch_size):
            if "domain" in samples:
                domains = np.array([samples["domain"][i]]).reshape(-1)
            elif pinfo:
                audio_id = samples["audio_id"][i]
                domains = pinfo[audio_id] 
            else:
                raise NotImplementedError(f"Find no prompting methods.")
                
            selected_domains = domains
            if use_random_selection:
                max_num_domains = int(np.random.uniform() * len(domains))        
                selected_domains = np.random.choice(domains, max_num_domains, replace=False) if max_num_domains > 0 else []

            prompt_str = prompt_format_func(selected_domains)
            prompt_strs.append(prompt_str)
    
    return prompt_strs


def transform_audio(samples,
                    use_null_inputs=False):
    batch_size = len(samples["audio"])
    out_samples = {}
    # Handle audio
    input_features = None
    if "input_features" in samples:
        input_features = samples["input_features"]
    elif use_null_inputs:
        input_features = np.zeros((batch_size, N_MELS, N_FRAMES))
    else:
        input_features = []
        for sample in samples["audio"]:
            padded_feat = whisper_feature_extractor(sample["array"].astype(np.float32)).numpy()
            input_features.append(padded_feat)
    out_samples["input_features"] = input_features
    
    return out_samples


def transform_for_prediction(samples,
                             use_cross_attn=True,
                             use_null_inputs=False,
                             use_prompts=True,
                             use_random_selection = True,
                             tokenizer: WhisperTokenizer = None,
                             pinfo: Union[List, Dict[str, List[str]]] = None):
    
    out_samples = {}
    if use_cross_attn:
        out_samples["input_features"] = transform_audio(samples, use_null_inputs=use_null_inputs)["input_features"]
    
    batch_size = len(samples["text"])
    # Handle labels and prompts
    if use_prompts:
        prompts = _get_prompts(samples, pinfo, use_random_selection)
    else:
        prompts = [''] * batch_size

    decoder_input_ids = []
    labels = []
    for text, prompt in zip(samples['text'], prompts):
        transcription_ids = tokenizer(preprocess_gigaspeech_text(text), add_special_tokens=True).input_ids # already add <SOT><EN><Tr><NT>...<EOT>
        prompt_ids = []
        if len(prompt):
            prompt_ids = [PREV_TOKEN_ID] + tokenizer(prompt, add_special_tokens=False).input_ids # <PREV> prompts
        
        # <PREV> prompts <SOT><EN><TR><NT> ... 
        one_decoder_input_ids = prompt_ids + transcription_ids[:-1]
        one_labels = [_CE_IGNORE_INDEX] * len(prompt_ids) + transcription_ids[1:]
        
        decoder_input_ids.append(one_decoder_input_ids)
        labels.append(one_labels)
        
        logging.debug(f"text: {text}; prompt: {prompt}")
        logging.debug(f"label: {one_labels}; \ndecoder_input_ids: {one_decoder_input_ids}")

    out_samples['decoder_input_ids'] = decoder_input_ids
    out_samples['labels'] = labels
    return out_samples


def transform_for_audio_text_generation(samples,
                                        use_prompts,
                                        use_random_selection,
                                        tokenizer: WhisperTokenizer = None,
                                        pinfo: Union[List, Dict[str, List[str]]] = None):
    assert 'audio' in samples
    out_samples = {}
    out_samples["text"] = samples["text"]
    out_samples["input_features"] = transform_audio(samples, use_null_inputs=False)["input_features"]
    batch_size = len(samples["text"])
    # Handle prompts
    if use_prompts:
        # <PREV>prompt<SOT><EN><TR><NT>
        prompts = _get_prompts(samples, pinfo, use_random_selection = use_random_selection)
        decoder_input_ids = []
        decoder_prompt_len = []
        for prompt in prompts:
            prompt_ids = [PREV_TOKEN_ID] + tokenizer(prompt, add_special_tokens=False).input_ids + tokenizer.prefix_tokens
            one_prompt_len = len(prompt_ids)
            decoder_input_ids.append(prompt_ids)
            decoder_prompt_len.append(one_prompt_len)
    else:
        # <SOT><EN><Tr><NT>
        decoder_input_ids = [tokenizer.prefix_tokens] * batch_size
        decoder_prompt_len = [len(tokenizer.prefix_tokens)] * batch_size

    out_samples["decoder_input_ids"] = decoder_input_ids
    out_samples["decoder_prompt_len"] = decoder_prompt_len
    return out_samples



def get_transform(cfg, tokenizer, stage='train'):
    if stage in {'train'}:
        pinfo_opt = {"train": "train_prompt_info"}[stage]
        pinfo = _load_pinfo(cfg[pinfo_opt])
        use_prompt_opt = {"train": "train_with_prompts"}[stage]
        logging.info(f"Get {stage} transform; use_prompt? {cfg[use_prompt_opt]}")
        return lambda x: transform_for_prediction(x,
                                                  use_cross_attn=cfg["use_cross_attn"],
                                                  use_null_inputs=cfg["use_null_inputs"],
                                                  use_prompts=cfg[use_prompt_opt],
                                                  use_random_selection=True,
                                                  tokenizer=tokenizer,
                                                  pinfo=pinfo)
    elif stage in {"validation", "test"}:
        pinfo_opt = {"test": "test_prompt_info", "validation": "valid_prompt_info"}[stage]
        pinfo = _load_pinfo(cfg[pinfo_opt])
        use_prompt_opt = {"test": "test_with_prompts", "validation": "valid_with_prompts"}[stage]
        logging.info(f"Get {stage} transform; use_prompt? {cfg[use_prompt_opt]}")
        return lambda x: transform_for_audio_text_generation(x,
                                                            use_prompts=cfg[use_prompt_opt],
                                                            use_random_selection=False,
                                                            tokenizer=tokenizer,
                                                            pinfo=pinfo
                                                            )

    raise NotImplementedError(f"{stage} stage not defined")