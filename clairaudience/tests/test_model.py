import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cur_dir}/../..')

from clairaudience.model import ClairaudienceForConditionalGeneration, bregman_div, ClairaudienceConfig, ClairaudienceModel, ClairaudienceDecoder

from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperDecoder


def _diff_model(model1, model2, keys_to_ignore={}):
    model1_params = {k: v for k, v in model1.named_parameters()}
    model2_params = {k: v for k, v in model2.named_parameters()}
    
    sum_diff = 0
    for n, p in model1_params.items():
        should_ignore = False
        for k in keys_to_ignore:
            if k in n:
                should_ignore = True
                break
        if should_ignore:
            continue
        
        sum_diff += th.square(model2_params[n] - p).sum()
    
    return sum_diff
        
    # sum_diff = th.tensor([th.square(model2_params[n] - p).sum() for n, p in model1_params.items()]).sum()
    
    # return sum_diff
    

def _fix_random_seed(seed=42):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.enabled = True


def test_decoder_only_forward():
    print(f"==== model decoder only forward (turned off encoder)")
    device = th.device('cuda:1')
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny.en", kl_coeff=0.1, use_cross_attn=False, use_no_speech_bias=True, use_kl_loss=True).to(device)

    batch_size, n_mel_ch, n_mel_feat = 8, 80, 3000
    vocab_size = model.config.vocab_size
    seq_len = 30
    max_source_len = 1

    input_features = th.rand((batch_size, n_mel_ch, n_mel_feat)).to(th.float32).to(device)
    labels = th.randint(0, vocab_size, (batch_size, seq_len)).to(th.long).to(device)
    encoder_last_hidden_state = th.zeros((batch_size, max_source_len, 384)).to(th.float32).to(device)

    inputs = dict(labels=labels)
    outputs = model(**inputs)

    print(f"model outputs keys = ", {k for k in outputs.keys()})
    print(f"encoder last hidden state = {outputs.encoder_last_hidden_state}")

    # th.testing.assert_close(encoder_last_hidden_state, outputs.encoder_last_hidden_state)
    assert outputs.encoder_last_hidden_state is None


def test_model_forward():
    device = th.device('cuda:1')
    
    kl_coeff = 0.1
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny.en", kl_coeff=kl_coeff, use_kl_loss=True).to(device)

    batch_size, n_mel_ch, n_mel_feat = 8, 80, 3000
    vocab_size = model.config.vocab_size
    seq_len = 30

    input_features = th.rand((batch_size, n_mel_ch, n_mel_feat)).to(th.float32).to(device)
    labels = th.randint(0, vocab_size, (batch_size, seq_len)).to(th.long).to(device)
    kl_target = th.rand((batch_size, seq_len, vocab_size)).to(th.float32).to(device)
    
    inputs = dict(input_features=input_features,
                  labels=labels)
    outputs = model(**inputs)

    print(f"model outputs keys = ", {k for k in outputs.keys()})
    print(f"encoder last hidden state = {outputs.encoder_last_hidden_state.shape}")

    lm_loss = outputs.lm_loss
    logits = outputs.logits
    kl_target = outputs.kl_target

    # Log KL
    kl_fct = th.nn.KLDivLoss(reduction='batchmean').to(device)
    def bg_fct(x, y): return bregman_div(x, y, 'batchmean')
    # log_logits = F.log_softmax(logits, dim=-1)
    print(f"==== Hand calculated kl")
    kl_loss = kl_fct(F.log_softmax(logits, -1), F.softmax(kl_target, -1))
    bg_loss = bg_fct(F.log_softmax(logits, -1), F.log_softmax(kl_target, -1))

    print(f"kl_loss = {kl_loss}; lm_loss = {lm_loss}l bg_loss = {bg_loss}")
    proj_loss = ((1 - kl_coeff) * lm_loss + kl_coeff * kl_loss).item()
    print(f"projected final loss with random target dist = {proj_loss}")

    # Pass old logits to model
    print(f"==== model calculated KL")

    print({k for k in outputs.keys()})
    print(f"loss = {outputs.loss}; lm_loss = {outputs.lm_loss}; kl_loss = {outputs.kl_loss}")

    # np.testing.assert_allclose(outputs.loss.item(), (outputs.lm_loss * (1 - kl_coeff) + kl_coeff * outputs.kl_loss).item())
    np.testing.assert_allclose(outputs.loss.item(), proj_loss)


def test_model_forward_backward(kl_type):
    device = th.device('cuda:1')
    kl_coeff = 0.1
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny.en", kl_coeff=kl_coeff, use_cross_attn=True, use_no_speech_bias=False, use_kl_loss=True).to(device)

    batch_size, n_mel_ch, n_mel_feat = 8, 80, 3000
    vocab_size = model.config.vocab_size
    seq_len = 30

    input_features = th.rand((batch_size, n_mel_ch, n_mel_feat)).to(th.float32).to(device)
    labels = th.randint(0, vocab_size, (batch_size, seq_len)).to(th.long).to(device)
    kl_target = th.rand((batch_size, seq_len, vocab_size)).to(th.float32).to(device)
    
    inputs = dict(input_features=input_features,
                  labels=labels)
    # Test backward
    optim = th.optim.Adam(model.parameters())
    optim.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"KL type = {kl_type}; loss = {loss.item()}; lm_loss = {outputs.lm_loss.item()} kl_loss = {outputs.kl_loss.item()}")
    loss.backward()
    optim.step()

def test_model_inference():
    device = th.device('cuda:1')
    kl_coeff = 0.1
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny.en", kl_coeff=kl_coeff, use_cross_attn=False, use_no_speech_bias=True, use_kl_loss=True).to(device)
    model.model.decoder.layers[0].no_speech_bias = th.nn.parameter.Parameter(th.ones(model.config.d_model, requires_grad=True, dtype=th.float32, device=device))
    
    batch_size, n_mel_ch, n_mel_feat = 8, 80, 3000
    vocab_size = model.config.vocab_size
    seq_len = 30

    input_features = th.rand((batch_size, n_mel_ch, n_mel_feat)).to(th.float32).to(device)
    labels = th.randint(0, vocab_size, (batch_size, seq_len)).to(th.long).to(device)
    kl_target = th.rand((batch_size, seq_len, vocab_size)).to(th.float32).to(device)
    
    inputs = dict(input_features=input_features,
                  labels=labels)
    
    outputs = model(**inputs)
    
    model.set_inference()
    outputs_infn = model(**inputs)
    
    assert th.square(outputs.logits - outputs_infn.logits).sum() > 0


def test_kl_div():
    inputs = F.softmax(th.rand(8, 10, 20), -1)
    target = F.softmax(th.rand_like(inputs), -1)

    reduction = 'batchmean'

    kl_func = th.nn.KLDivLoss(reduction=reduction)
    kl_loss = kl_func(th.log(inputs), target)

    bg_loss = bregman_div(th.log(inputs), th.log(target), reduction=reduction)

    th.testing.assert_close(kl_loss, bg_loss)


def test_model_loading():
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    
    cl_model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny",use_no_speech_bias=True, use_cross_attn=False, use_kl_loss=True)

    # Check loading
    assert cl_model.model.__class__.__name__ == 'ClairaudienceModel', cl_model.model.__class__.__name__
    assert cl_model.model.decoder.__class__.__name__ == 'ClairaudienceDecoder', cl_model.model.decoder.__class__.__name__
    assert cl_model.model.target_decoder.__class__.__name__ == 'WhisperDecoder', cl_model.model.target_decoder.__class__.__name__
    
    # Expect that the freshly initialized decoder model having weight the same as the whisper tiny model
    keys_to_ignore = {"no_speech_bias"}
    diff_decoder = _diff_model(cl_model.model.decoder, whisper_model.model.decoder, keys_to_ignore)
    assert diff_decoder == 0, f"{diff_decoder}"
    assert hasattr(cl_model.model.decoder.layers[0], "no_speech_bias")
    
    diff_decoder = _diff_model(cl_model.model.target_decoder, whisper_model.model.decoder)
    assert diff_decoder == 0, f"{diff_decoder}"
    
    # Set the target decoder the same as the decoder
    cl_model.model.target_decoder.load_state_dict(cl_model.model.decoder.state_dict(), strict=False)
    cl_model.save_pretrained("outputs/test_model_loading.pth")
    
    # Load the model from pretrained
    new_model_config = ClairaudienceConfig.from_pretrained("outputs/test_model_loading.pth")
    new_cl_model = ClairaudienceForConditionalGeneration.from_pretrained("outputs/test_model_loading.pth")
    keys_to_ignore = {"no_speech_bias"}
    diff_decoder = _diff_model(new_cl_model.model.decoder, whisper_model.model.decoder, keys_to_ignore)
    assert diff_decoder == 0, f"{diff_decoder}"
    diff_decoder = _diff_model(new_cl_model.model.target_decoder, whisper_model.model.decoder, keys_to_ignore)
    assert diff_decoder == 0, f"{diff_decoder}"
    
    assert new_model_config.use_no_speech_bias == True
    assert new_model_config.use_cross_attn == False
    assert new_model_config.use_kl_loss == True
    
    # Check loading
    assert new_cl_model.model.__class__.__name__ == 'ClairaudienceModel', new_cl_model.model.__class__.__name__
    assert new_cl_model.model.decoder.__class__.__name__ == 'ClairaudienceDecoder', new_cl_model.model.decoder.__class__.__name__
    assert new_cl_model.model.target_decoder.__class__.__name__ == 'WhisperDecoder', new_cl_model.model.target_decoder.__class__.__name__
    

def test_model_freezing():
    device = th.device("cuda:1")
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny").to(device)
    model.freeze_encoder()
    
    batch_size, n_mel_ch, n_mel_feat = 8, 80, 3000
    vocab_size = model.config.vocab_size
    seq_len = 30
    input_features = th.rand((batch_size, n_mel_ch, n_mel_feat)).to(th.float32).to(device)
    labels = th.randint(0, vocab_size, (batch_size, seq_len)).to(th.long).to(device)
    inputs = dict(input_features=input_features,
                  labels=labels)
    # Test backward
    optim = th.optim.Adam(model.parameters())
    optim.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"loss = {loss.item()}; lm_loss = {outputs.lm_loss.item()} kl_loss = {outputs.kl_loss.item()}")
    loss.backward()
    optim.step()
    
    old_model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny").to(device)
    
    diff_sum = _diff_model(old_model.model.encoder, model.model.encoder)
    
    assert diff_sum == 0

@th.no_grad()
def test_decoder_position_shift():
    device = th.device('cuda:1')
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny")
    decoder = ClairaudienceDecoder(model.config)
    bz = 10
    seq_len = 20
    d_model = 300
    for _ in range(10):
        positions = th.rand(seq_len, d_model).to(device)
        shifts = th.randint(low = 0, high = 10, size = (10,))
        attention_mask = th.ones(bz, seq_len).to(device)
        for i, s in enumerate(shifts):
            for j in range(s):
                attention_mask[i,j] = 0
        shifted_positions = decoder.shift_by_attention_mask(attention_mask, positions)
        for i, s in enumerate(shifts):
            for j in range(0, seq_len-s):
                assert (positions[j,:] == shifted_positions[i,j+s,:]).all()

@th.no_grad()
def test_decoder_position_select():
    device = th.device('cuda:1')
    model = ClairaudienceForConditionalGeneration.from_whisper_pretrained("openai/whisper-tiny")
    decoder = ClairaudienceDecoder(model.config)
    bz = 10
    seq_len = 20
    d_model = 300
    for _ in range(10):
        positions = th.rand(seq_len, d_model).to(device)
        shifts = th.randint(low = 0, high = 10, size = (10,))
        attention_mask = th.ones(bz, seq_len).to(device)
        for i, s in enumerate(shifts):
            for j in range(s):
                attention_mask[i,j] = 0
        shifted_positions = decoder.select_by_attention_mask(attention_mask, positions)
        for i, s in enumerate(shifts):
            assert (positions[seq_len-s-1,:] == shifted_positions[i,0,:]).all()



if __name__ == '__main__':
    _fix_random_seed()
    test_model_forward()
    test_kl_div()
    test_model_forward_backward('KL_div')
    test_model_forward_backward('Bregman_div')
    test_decoder_only_forward()
    
    test_model_loading()
    test_model_inference()
    
    test_model_freezing()
    
    test_decoder_position_shift()
    test_decoder_position_select()