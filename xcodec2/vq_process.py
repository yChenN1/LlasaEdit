import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder_vocos import CodecDecoderVocos
from vq.module import SemanticEncoder
from collections import OrderedDict

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

def load_models(ckpt_path):
    print(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']

    def filter_state_dict(prefix):
        return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

    encoder = CodecEncoder()
    encoder.load_state_dict(filter_state_dict('CodecEnc.'))
    encoder.to(device).eval()
    
    decoder = CodecDecoderVocos()
    decoder.load_state_dict(filter_state_dict('generator.'))
    decoder.to(device).eval()

    semantic_encoder = SemanticEncoder(1024, 1024, 1024)
    semantic_encoder.load_state_dict(filter_state_dict('SemanticEncoder_module.'))
    semantic_encoder.to(device).eval()

    fc_post_a = nn.Linear(2048, 1024)
    fc_post_a.load_state_dict(filter_state_dict('fc_post_a.'))
    fc_post_a.to(device).eval()

    fc_prior = nn.Linear(2048, 2048)
    fc_prior.load_state_dict(filter_state_dict('fc_prior.'))
    fc_prior.to(device).eval()

    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)
    semantic_model.to(device).eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    return encoder, decoder, semantic_encoder, fc_post_a, fc_prior, semantic_model, feature_extractor


models = load_models('/mnt/fast/nobackup/scratch4weeks/yc01815/pretrain_models/xcodec2/xcodec2.ckpt')
encoder, decoder, semantic_encoder, fc_post_a, fc_prior, semantic_model, feature_extractor = models

def extract_vq_code(wav, encoder=encoder, decoder=decoder, semantic_model=semantic_model, semantic_encoder=semantic_encoder, fc_prior=fc_prior, feature_extractor=feature_extractor, sr=16000):
   
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).cuda()
    wav_tensor = F.pad(wav_tensor, (0, 320 - wav_tensor.shape[1] % 320))
    # __import__('pdb').set_trace()
    
    feat = feature_extractor(F.pad(wav_tensor[0].cpu(), (160, 160)), sampling_rate=sr, return_tensors="pt")['input_features'].cuda()

    with torch.no_grad():
        vq_emb = encoder(wav_tensor.unsqueeze(1)).transpose(1, 2)
        semantic_target = semantic_model(feat).hidden_states[16].transpose(1, 2)
        semantic_target = semantic_encoder(semantic_target)
        vq_input = torch.cat([semantic_target, vq_emb], dim=1)
        vq_input = fc_prior(vq_input.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = decoder(vq_input, vq=True)

    return vq_code


def extract_vq_code_for_offline_training(input_waveform, input_features, encoder=encoder, decoder=decoder, semantic_model=semantic_model, semantic_encoder=semantic_encoder, fc_prior=fc_prior, sr=16000):
    feat = input_features
    with torch.no_grad():
        vq_emb = encoder(input_waveform).transpose(1, 2)   # input_wavform: [B, 1, length]
        semantic_target = semantic_model(feat.squeeze(1)).hidden_states[16].transpose(1, 2)
        semantic_target = semantic_encoder(semantic_target)
        vq_input = torch.cat([semantic_target, vq_emb], dim=1)
        vq_input = fc_prior(vq_input.transpose(1, 2)).transpose(1, 2)
        _, vq_code, _ = decoder(vq_input, vq=True)

    return vq_code.squeeze(1)


def reconstruct_from_vq_code(vq_code, decoder=decoder, fc_post_a=fc_post_a):
    with torch.no_grad():
        vq_post_emb = decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2)).transpose(1, 2)
        vq_post_emb = fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon = decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze().cpu().numpy()
    return recon



# wav, _ = librosa.load('/mnt/fast/nobackup/scratch4weeks/yc01815/ears/p001/emo_adoration_freeform.wav', sr=16000)
# vq_code = extract_vq_code(wav, encoder, decoder, semantic_model, semantic_encoder, fc_prior, feature_extractor, 16000)
# recon_audio = reconstruct_from_vq_code(vq_code, decoder, fc_post_a)
# sf.write('./abc.wav', recon_audio, 16000)