import os
import sys
import jiwer
import pandas as pd
import numpy as np
import logging
from git import Repo
from whisper.normalizers import EnglishTextNormalizer
from datasets import Dataset
from rich import print

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{cur_dir}/..")

from clairaudience.data_process import preprocess_gigaspeech_text

def get_repo_hash(file_path, log_dir) -> str:
    repo = Repo(file_path, search_parent_directories=True)
    sha = repo.head.object.hexsha

    with open(f"{log_dir}/sha.txt", "w+") as f:
        f.write(f"{sha}\n")

    return sha


def extract_word_error_info(tgts, preds):
    outputs = jiwer.process_words(tgts, preds)
    align_dict = dict(insert_pred=[], substitute_tgt=[], substitute_pred=[], delete_tgt=[], delete_pred=[])
    
    num_sen = len(outputs.references)
    for i in range(num_sen):
        ref = outputs.references[i]
        hyp = outputs.hypotheses[i]
        one_align_dict = dict(insert_pred=[], substitute_tgt=[], substitute_pred=[], delete_tgt=[], delete_pred=[])
        for ac in outputs.alignments[i]:
            if ac.type == 'equal':
                continue
            tgt_words = ref[ac.ref_start_idx: ac.ref_end_idx]
            pred_words = hyp[ac.hyp_start_idx: ac.hyp_end_idx]
            one_align_dict[f"{ac.type}_pred"].extend(pred_words) 
            if ac.type != 'insert':
                one_align_dict[f"{ac.type}_tgt"].extend(tgt_words)
        for k, v in one_align_dict.items():
            align_dict[k].append(v)
            
    align_sen = np.array(jiwer.visualize_alignment(outputs, show_measures=False).split('\n')).reshape(-1, 5)
    align_strs = ['\n'.join(sen) for sen in align_sen]
    return dict(viz_align=align_strs, **align_dict)


def analyzed_raw_data(dataset: Dataset, output_dir: str = None, target_column: str = 'target', pred_column: str = 'pred', normalizer = None):
    targets = dataset[target_column]
    preds = dataset[pred_column]
    if normalizer is None:
        whisper_normalizer = EnglishTextNormalizer()
        normalizer = lambda x: whisper_normalizer(preprocess_gigaspeech_text(x))
    targets_clean = [normalizer(text) for text in targets]
    preds_clean = [normalizer(text) for text in preds]
    wers = []
    targets_filter = []
    preds_filter = []
    clean_ids = []
    
    targets_null_type =[]
    preds_null_type =[]
    
    for i, (ref, hyp) in enumerate(zip(targets_clean, preds_clean)):
        try:
            if len(ref):
                wers.append(jiwer.wer(ref, hyp))
                clean_ids.append(i)
                targets_filter.append(ref)
                preds_filter.append(hyp)
            else:
                targets_null_type.append(ref)
                preds_null_type.append(hyp)
        except ValueError as e:
            print(f"Skip a row as {getattr(e, 'message', repr(e))} occurs in target: {ref}")
    wer = jiwer.wer(targets_filter, preds_filter)
    logging.info(f'WER = {wer}')
    we_info = extract_word_error_info(targets_filter, preds_filter)
    pd_data = pd.DataFrame(dict(wer=wers, **we_info, 
                                target_clean=targets_filter,
                                pred_clean=preds_filter,
                                target=np.array(targets)[clean_ids],
                                pred=np.array(preds)[clean_ids]))
    # No caption error
    no_caption_error = np.sum([len(p) != 0 for p in preds_null_type]) / len(targets_null_type) if len(targets_null_type) else np.nan
    no_caption_pd_data = pd.DataFrame(dict(target=targets_null_type, pred=preds_null_type))
    logging.info(f"No caption error = {no_caption_error}") 
    
    metrics = dict(wer=wer, no_caption_error=no_caption_error)
    outputs = dict(wer_pd_data=pd_data, no_caption_error_pd_data=no_caption_pd_data, metrics=metrics)
    return outputs
