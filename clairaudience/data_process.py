import os
import sys
import torch as th
import numpy as np
from datasets import DatasetDict, load_from_disk, Audio
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import logging
import re
import nltk
import truecase
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{cur_dir}/..")

# NOTE: ACP only config
nltk.data.path.append("/proj/gpu_d_98001/proj_asr/nltk")

from clairaudience.model import whisper_feature_extractor


def preprocess_gigaspeech_text(text):
    text = re.sub(string = text, pattern = " <COMMA>", repl = ",")
    text = re.sub(string = text, pattern = " <PERIOD>", repl = ".")
    text = re.sub(string = text, pattern = " <EXCLAMATIONPOINT>", repl = "!")
    text = re.sub(string = text, pattern = " <QUESTIONMARK>", repl = "?")
    text = re.sub(string = text, pattern = "<\w*>", repl = "")
    text = truecase.get_true_case(text)
    text = re.sub(r'\s+\.', '.', text)
    return text


def prepare_gigaspeech_dataset(batch, feature_extractor = whisper_feature_extractor, tokenizer: WhisperTokenizer = None):
    batch = default_prepare_dataset(batch, feature_extractor, None)
    
    if tokenizer is not None:
        batch["labels"] = tokenizer(preprocess_gigaspeech_text(batch["text"])).input_ids
    return batch
    

def default_prepare_dataset(batch, feature_extractor = whisper_feature_extractor, tokenizer: WhisperTokenizer = None):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    # batch["input_features"] = feature_extractor(audio["array"], sampling_rate=feature_extractor.sampling_rate).input_features[0]
    batch["input_features"] = whisper_feature_extractor(audio["array"])
    
    if tokenizer is not None:
        # NOTE: special tokens are added
        batch["labels"] = tokenizer(batch["text"]).input_ids

    return batch


def prepare_imdb_dataset(batch, feature_extractor = whisper_feature_extractor, tokenizer: WhisperTokenizer = None):
    # Dummy audio
    audio = dict(array=np.zeros((3000, )), sampling_rate=16000)
    # batch["input_features"] = feature_extractor(audio["array"], sampling_rate=feature_extractor.sampling_rate).input_features[0]
    batch["input_features"] = whisper_feature_extractor(audio["array"])

    # Training forma: <EN> $current_labels <EOT>
    # Whisper max decoder input len = 448
    max_len = 448  # TODO: load this from somewhere
    labels = tokenizer(batch["text"], add_special_tokens=False).input_ids
    if len(labels) > max_len - 2:
        labels = labels[:max_len - 2]
    batch["labels"] = [50259, *labels, 50257]
    # TODO: to be removed
    if len(batch['labels']) > max_len:
        print(len(batch["labels"]), tokenizer.decode(batch["labels"], add_special_tokens=False))
    return batch


def filter_tedlium(target_text):
    return target_text != "ignore_time_segment_in_scoring"


def setup_dataset(dataset_name, 
                  dataset_path, 
                  feature_extractor: WhisperFeatureExtractor = None, 
                  tokenizer: WhisperTokenizer = None,
                  num_subsamples: int = -1,
                  cache_file_name="~/.cache/clairaudience/dataset.cache", **kwargs) -> DatasetDict:
    dataset = load_from_disk(dataset_path)

    dataset_pre_func = lambda x: default_prepare_dataset(x, feature_extractor, tokenizer)
    splits = ["train", "test"]
    
    if dataset_name in {"common_voice_5_1", "common_voice_9_0"}:
        dataset = dataset.select_columns(['audio', 'sentence']).rename_columns({'audio': 'audio', 'sentence': 'text'})
    elif dataset_name == 'imdb_movie_reviews':
        dataset = dataset.select_columns(['text'])
        dataset_pre_func = lambda x: prepare_imdb_dataset(x, feature_extractor, tokenizer)
    elif dataset_name == 'gigaspeech':
        dataset = dataset.select_columns(['audio', 'text', 'audio_id'])
        dataset_pre_func = None
        splits = ["train", "validation", "test"]
        # HACK: use validation as test set
        # dataset["test"] = dataset["validation"]
        # del dataset["validation"]
    elif dataset_name == 'gigaspeech_extracted':
        dataset = dataset.select_columns(['text', 'input_features', 'domain'])
        splits = ["train", "validation", "test"]
        # HACK: use validation as test set
        dataset["test"] = dataset["validation"]
        # del dataset["validation"]
    elif dataset_name == "tedlium":
        dataset = dataset.select_columns(['audio', 'text'])
        # Filter out ignore segments
        dataset = dataset.filter(filter_tedlium, input_columns=["text"], cache_file_name=f"{cache_file_name}.tedlium_filter.cache")
    elif dataset_name in {"aim_ihm", "aim_sdm"}:
        dataset = dataset.select_columns(['audio', 'text'])
    elif dataset_name == "librispeech_asr":
        dataset = dataset.select_columns(['audio', 'text'])
    elif dataset_name == "gigaspeech":
        dataset = dataset.select_columns(['audio', 'text'])
    elif dataset_name == "spgispeech":
        dataset_pre_func = None
        splits = ["test"]
        dataset = dataset.select_columns(['audio', 'transcript']).rename_columns({'audio': 'audio', 'transcript': 'text'})
    elif dataset_name == "earnings22":
        dataset = dataset.select_columns(['audio', 'sentence']).rename_columns({'audio': 'audio', 'sentence': 'text'})
    elif dataset_name == "voxpopuli":
        dataset = dataset.select_columns(['audio', 'normalized_text']).rename_columns({'audio': 'audio', 'normalized_text': 'text'})
    elif dataset_name == "atc":
        dataset_pre_func = None
        splits = ["test"]
        dataset = dataset.select_columns(['audio', 'text'])
    elif dataset_name == "atco2":
        dataset_pre_func = None
        dataset = dataset.select_columns(['audio', 'text', 'domain'])
        # HACK: put train to test
        dataset = DatasetDict(dict(test=dataset['train']))
        splits = ["test"]
    elif dataset_name == 'kaggle_medical':
        dataset_pre_func = None
        dataset = dataset.select_columns(['audio', 'phrase', 'prompt']).rename_columns(dict(audio='audio', phrase='text', prompt='domain'))
        # HACK: Put the medical dataset into DatasetDict format
        splits = ["test"]
        dataset = DatasetDict(dict(test=dataset))
    else:
        raise NotImplementedError(f"dataset name: {dataset_name} not recognizable.")
    
    cache_file_names = {split: cache_file_name for split in splits}
    dataset = DatasetDict({split: dataset[split] for split in splits})
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if num_subsamples > 0:
        dataset = _subsample_datasets(dataset, num_subsamples, num_subsamples, num_subsamples)
    else:
        dataset = _subsample_datasets(dataset, 
                                      kwargs["num_train_subsamples"],
                                      kwargs["num_valid_subsamples"],
                                      kwargs["num_test_subsamples"])
    if dataset_pre_func:
        dataset = dataset.map(dataset_pre_func, num_proc=8, cache_file_names=cache_file_names)

    return dataset

def _subsample_datasets(datasets, num_train_subsamples, num_valid_subsamples, num_test_subsamples):
    name2nsamples = {'train': num_train_subsamples, 'validation': num_valid_subsamples, 'test': num_test_subsamples}
    for k in datasets.keys():
        num_samples = len(datasets[k])
        num_subsamples = name2nsamples[k]
        if num_subsamples < 0:
            continue
        selected_ids = np.random.choice(num_samples, size=(num_subsamples,), replace=False)
        datasets[k] = datasets[k].select(selected_ids)
        logging.info(f"Subsample the {k} dataset. Only use {num_subsamples} samples of the {k} dataset ({num_samples})")
    return datasets


def warp_hf_dataset_to_th_dataset(dataset):
    class WarpDataset(th.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __getitem__(self, idx):
            return self.dataset[idx]
        
        def __len__(self):
            return len(self.dataset)
        
    return WarpDataset(dataset)
        