import os
import sys
from rich import print
import numpy as np
import torch as th
import torch.nn.functional as F
from transformers import WhisperTokenizer, WhisperProcessor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cur_dir}/../..')

from clairaudience.data_transform import transform_audio, transform_for_audio_text_generation, transform_for_prediction, _get_prompts, DataCollatorSpeechSeq2SeqWithPadding, prompt_format_func


def test_prompt_format_func():
    domains = ["duck", "egg", "chicken"]
    gt = "[ domain: duck, egg, chicken]"
    prompt = prompt_format_func(domains)
    assert gt == prompt, f"{prompt}; {gt}"


def test_transform():
    samples = {
        'audio': [{"array": np.random.rand(80, 3000)}],
        'text': ["My name is clairaudience"],
        'audio_id': ['123']
    }
    pinfo = {'123': ['duck', 'egg']}
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="english", task="transcribe")
    outputs = transform_for_audio_text_generation(samples, use_prompts=True, use_random_selection=False, tokenizer=tokenizer, pinfo=pinfo)
    gt_labels = [[-100, -100, -100, -100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834, 50257]]
    gt_inputs = [[58, 9274, 25, 12482, 11, 3777, 60, 50258, 50259, 50359, 50363]]
    
    labels = outputs["labels"]
    inputs = outputs["decoder_input_ids"]
    np.testing.assert_allclose(gt_labels, labels)
    np.testing.assert_allclose(gt_inputs, inputs)
    print(f"labels            = {tokenizer.decode(labels[0])}")
    print(f"decoder_input_ids = {tokenizer.decode(inputs[0])}")
    
    pinfo=["We like youtube videos"]
    outputs = transform_for_prediction(samples, use_prompts=True, use_null_inputs=False, use_random_selection=False, tokenizer=tokenizer, pinfo=pinfo)
    gt_labels = [[-100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834, 50257]]
    gt_inputs = [[4360, 411, 12487, 2145, 50258, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834]]
    labels = outputs["labels"]
    inputs = outputs["decoder_input_ids"]
    np.testing.assert_allclose(gt_labels, labels)
    np.testing.assert_allclose(gt_inputs, inputs)
    print(f"labels            = {tokenizer.decode(labels[0])}")
    print(f"decoder_input_ids = {tokenizer.decode(inputs[0])}")

    assert len(labels[0]) == len(inputs[0])

def test_collate_fn():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="english", task="transcribe")
    
    samples = [{"input_features": np.random.rand(80, 3000).astype(np.float32),
                "decoder_input_ids": [58, 9274, 25, 12482, 11, 3777, 60, 50258, 50259, 50359, 50363], 
                "labels": [-100, -100, -100, -100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834, 50257]}] * 4
    samples = [{"input_features": np.random.rand(80, 3000).astype(np.float32),
                "decoder_input_ids": [58, 9274, 25, 12482, 11, 3777, 60, 50258, 50259, 50359, 50363], 
                "labels": [-100, -100, -100, -100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 308, 41375, 4834, 50257]}] + samples
    
    collat_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, generation_mode=True)
    
    outputs = collat_fn(samples)
    labels = outputs["labels"]
    inputs = outputs["decoder_input_ids"]
    assert np.allclose(labels.shape, [5, 17]), f"{labels.shape}"
    assert np.allclose(inputs.shape, [5, 11]), f"{inputs.shape}"
    # id 1 - 4 are padded with -100
    assert np.allclose(labels[1:, -2:], [50257, -100]), f"{labels[1:, -2:]}"
    
    
    samples = [{"input_features": np.random.rand(80, 3000).astype(np.float32),
                "decoder_input_ids": [58, 9274, 25, 12482, 11, 3777, 60, 50258, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834], 
                "labels": [-100, -100, -100, -100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 41375, 4834, 50257]}] * 4
    samples = [{"input_features": np.random.rand(80, 3000).astype(np.float32),
                "decoder_input_ids": [58, 9274, 25, 12482, 11, 3777, 60, 50258, 50259, 50359, 50363, 2226, 1315, 307, 308, 41375, 4834], 
                "labels": [-100, -100, -100, -100, -100, -100, -100, 50259, 50359, 50363, 2226, 1315, 307, 308, 41375, 4834, 50257]}] + samples
    collat_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, generation_mode=False)
    
    outputs = collat_fn(samples)
    labels = outputs["labels"]
    inputs = outputs["decoder_input_ids"]
    assert np.allclose(labels.shape, inputs.shape)
    assert np.allclose(labels.shape, [5, 17]), f"{labels.shape}"
    assert np.allclose(inputs.shape, [5, 17]), f"{inputs.shape}"
    # id 1 - 4 are padded with -100
    assert np.allclose(labels[1:, -2:], [50257, -100]), f"{labels[1:, -2:]}"
    # id 1 - 4 are padded with EOT
    assert np.allclose(inputs[1:, -2:], [4834, 50257]), f"{inputs[1:, -2:]}"


if __name__ == '__main__':
    test_prompt_format_func()
    test_transform()
    test_collate_fn()