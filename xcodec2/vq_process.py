# import os
# import librosa
# import torch
# import torch.nn.functional as F
# import numpy as np
# import soundfile as sf
# from glob import glob
# from tqdm import tqdm
# from os.path import basename, join, exists
# from vq.codec_encoder import CodecEncoder
# # from vq.codec_decoder import CodecDecoder
# from vq.codec_decoder_vocos import CodecDecoderVocos
# from argparse import ArgumentParser
# from time import time
# from transformers import  AutoModel
# import torch.nn as nn
# from vq.module import SemanticDecoder,SemanticEncoder
# from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--input-dir', type=str, default='test_audio/input_test')
#     parser.add_argument('--ckpt', type=str, default='/mnt/fast/nobackup/scratch4weeks/yc01815/pretrain_models/xcodec2/xcodec2.ckpt')
#     parser.add_argument('--output-dir', type=str, default='test_audio/output_test')
             
#     args = parser.parse_args()
#     sr = 16000

#     print(f'Load codec ckpt from {args.ckpt}')
#     ckpt = torch.load(args.ckpt, map_location='cpu')
#     ckpt=ckpt['state_dict']

#     state_dict = ckpt
#     from collections import OrderedDict
#     # 步骤 2：提取并过滤 'codec_enc' 和 'generator' 部分
#     filtered_state_dict_codec = OrderedDict()
#     filtered_state_dict_semantic_encoder = OrderedDict()
#     filtered_state_dict_gen = OrderedDict()
#     filtered_state_dict_fc_post_a = OrderedDict()
#     filtered_state_dict_fc_prior = OrderedDict()
#     for key, value in state_dict.items():
#         if key.startswith('CodecEnc.'):
#             # 去掉 'codec_enc.' 前缀，以匹配 CodecEncoder 的参数名
#             new_key = key[len('CodecEnc.'):]
#             filtered_state_dict_codec[new_key] = value
#         elif key.startswith('generator.'):
#             # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
#             new_key = key[len('generator.'):]
#             filtered_state_dict_gen[new_key] = value
#         elif key.startswith('fc_post_a.'):
#             # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
#             new_key = key[len('fc_post_a.'):]
#             filtered_state_dict_fc_post_a[new_key] = value
#         elif key.startswith('SemanticEncoder_module.'):
#             # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
#             new_key = key[len('SemanticEncoder_module.'):]
#             filtered_state_dict_semantic_encoder[new_key] = value
#         elif key.startswith('fc_prior.'):
#             # 去掉 'generator.' 前缀，以匹配 CodecDecoder 的参数名
#             new_key = key[len('fc_prior.'):]
#             filtered_state_dict_fc_prior[new_key] = value
#     semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)

#     semantic_model.eval().cuda()
#     SemanticEncoder_module = SemanticEncoder(1024,1024,1024)
#     SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
#     SemanticEncoder_module = SemanticEncoder_module.eval().cuda()
#     encoder = CodecEncoder()
#     encoder.load_state_dict(filtered_state_dict_codec)
#     encoder = encoder.eval().cuda()
#     decoder = CodecDecoderVocos()
#     decoder.load_state_dict(filtered_state_dict_gen)
#     decoder = decoder.eval().cuda()
#     fc_post_a = nn.Linear( 2048, 1024 )
#     fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
#     fc_post_a = fc_post_a.eval().cuda()
#     fc_prior = nn.Linear( 2048, 2048 )
#     fc_prior.load_state_dict(filtered_state_dict_fc_prior)
#     fc_prior = fc_prior.eval().cuda()
#     feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

#     wav_dir = args.output_dir
#     os.makedirs(wav_dir, exist_ok=True)

    
    
#     # wav_paths = glob(join(args.input_dir, '*.flac')) #
#     # both wav and flac and mp3
#     wav_paths = glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
#     flac_paths = glob(os.path.join(args.input_dir, '**', '*.flac'), recursive=True)
#     mp3_paths = glob(os.path.join(args.input_dir, '**', '*.mp3'), recursive=True)

#     # 合并所有路径
#     wav_paths = wav_paths + flac_paths + mp3_paths
#     print(f'Found {len(wav_paths)} wavs in {args.input_dir}')
    
#     st = time()
#     for wav_path in tqdm(wav_paths):
#         # __import__('pdb').set_trace()
#         target_wav_path = join(wav_dir, basename(wav_path))
#         wav = librosa.load(wav_path, sr=sr)[0] 
#         wav_cpu = torch.from_numpy(wav)

 
#         wav = wav_cpu.unsqueeze(0).cuda()
#         pad_for_wav = (320 - (wav.shape[1] % 320))
 
#         wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

#         feat =  feature_extractor(F.pad(wav[0,:].cpu(), (160, 160)), sampling_rate=16000, return_tensors="pt") .data['input_features']
        
#         feat = feat.cuda()

#         with torch.no_grad():
#             vq_emb = encoder(wav.unsqueeze(1))
#             vq_emb = vq_emb.transpose(1, 2)

 
#             semantic_target = semantic_model(feat[:,  :,:])

#             semantic_target = semantic_target.hidden_states[16]

#             semantic_target = semantic_target.transpose(1, 2)
#             semantic_target = SemanticEncoder_module(semantic_target)
             

#             vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
#             vq_emb =  fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

#             _, vq_code, _ = decoder(vq_emb, vq=True)  # vq_code here !!!!

#             vq_post_emb = decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
#             vq_post_emb = vq_post_emb.transpose(1, 2)
#             vq_post_emb = fc_post_a(vq_post_emb.transpose(1,2)).transpose(1,2)
#             recon = decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze().detach().cpu().numpy()
#             # recon = decoder(decoder.vq2emb(vq_code.transpose(1,2)).transpose(1,2), vq=False).squeeze().detach().cpu().numpy()
#         sf.write(target_wav_path, recon, sr)
#     et = time()
#     print(f'Inference ends, time: {(et-st)/60:.2f} mins')

# assert False

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


def reconstruct_from_vq_code(vq_code, decoder=decoder, fc_post_a=fc_post_a):
    with torch.no_grad():
        vq_post_emb = decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2)).transpose(1, 2)
        vq_post_emb = fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon = decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze().cpu().numpy()
    return recon



# wav, _ = librosa.load('/mnt/fast/nobackup/users/yc01815/code/X-Codec-2.0/test_audio/input_test/test.flac', sr=16000)
# vq_code = extract_vq_code(wav, encoder, decoder, semantic_model, semantic_encoder, fc_prior, feature_extractor, 16000)
# recon_audio = reconstruct_from_vq_code(vq_code, decoder, fc_post_a)
# sf.write('./abc.wav', recon_audio, 16000)