from math import exp
import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Resample
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor
sys.path.append('/mnt/fast/nobackup/users/jz01101/cy/LlasaEdit/xcodec2')
from vq_process import load_models, extract_vq_code, reconstruct_from_vq_code

import warnings
warnings.filterwarnings("ignore")


# Set CUDA-related environment variables
os.environ["CUDA_HOME"] = os.path.expanduser("~/cuda-12.6")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Set GCC compiler paths from conda
conda_prefix = os.environ.get("CONDA_PREFIX", "/path/to/your/conda/env")  # fallback if not in conda env
os.environ["CC"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-gcc"
os.environ["CXX"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-g++"
os.environ["LD_LIBRARY_PATH"] = (
    f"{conda_prefix}/lib:"
    f"{conda_prefix}/x86_64-conda-linux-gnu/lib:"
    f"{conda_prefix}/lib/gcc/x86_64-conda-linux-gnu/12.4.0:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)


# === Load Models ===
llasa_1b = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/qic/0808_a2a_0807_lora_mlp_rank64alpha128_lr1e4_iter8000'
# llasa_1b ='HKUSTAudio/Llasa-1B'
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
llm_model = AutoModelForCausalLM.from_pretrained(llasa_1b).eval().cuda()


# === Helper Functions ===
def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            speech_ids.append(int(token_str[4:-2]))
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def ids_to_tokens(speech_ids): 
    return [f"<|s_{i}|>" for i in speech_ids]

def replace_tagged_token(token_list, target_token, new_sequence):
    idx = token_list.index(target_token)
    return token_list[:idx] + list(new_sequence) + token_list[idx+1:]


def replace_tagged_token_torch(token_tensor: torch.Tensor, target_token: int, new_sequence: torch.Tensor) -> torch.Tensor:
    """
    在 token_tensor 中找到 target_token，把它替换为 new_sequence（可以是多个 token）。

    Args:
        token_tensor (torch.Tensor): 1D tensor (e.g., torch.int64)
        target_token (int): token ID to replace
        new_sequence (torch.Tensor): 1D tensor of new token IDs

    Returns:
        torch.Tensor: new 1D tensor after replacement
    """
    # __import__('pdb').set_trace()
    if token_tensor.dim() != 1 or new_sequence.dim() != 1:
        raise ValueError("Both input tensors must be 1D")

    device = token_tensor.device
    idx = (token_tensor == target_token).nonzero(as_tuple=False).squeeze()

    if idx.numel() == 0:
        raise ValueError(f"Token {target_token} not found in tensor.")

    # If multiple matches, take first only
    idx = idx.item() if idx.ndim == 0 else idx[0].item()

    return torch.cat([token_tensor[:idx], new_sequence.to(device), token_tensor[idx+1:]])

use_text = False
text_front = True

speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
eos_token_id = speech_generation_end_id
if not use_text or text_front:
    eos_token_id = speech_generation_end_id
else:
    eos_token_id = text_generation_end_id


# === Input: eval audio set ===
test_data_split = load_dataset(
        'parquet',
        data_files={
            'train': [
                '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks_valid/chunk_valid.parquet',
            ]
        },
        split='train',
    )


# split = 'valid'
# audio_eval_path = f'/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/{split}_instruct.csv'
# eval_list = pd.read_csv(audio_eval_path).to_dict(orient='records')[:]
# base_path = f'/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_{split}'
# 获取当前日期
today = datetime.now().strftime("%Y%m%d")

# 提取步数，比如 'checkpoint-100000' → '10000'
basename = os.path.basename(llasa_1b)
if basename.startswith("checkpoint-"):
    step = basename.replace("checkpoint-", "")[:5]
else:
    step = "unknown"

# 判断是否是 lora 模式
if "finetune_lora" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/lora/{today}/{step}"
elif 'grpo' in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/grpo/{today}/{step}"
elif "finetune" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/jz01101/llasa/evaluation/{today}/{step}_tempo0.95_sad"
elif "text" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/text/{today}/{step}"
else:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/jz01101/llasa/evaluation/other/{today}/{step}2"

print("Save path:", save_path)
os.makedirs(save_path, exist_ok=True)


# def process_audio(audio_array, sr, target_sr):
#     audio_norm_scale = 1.0
#     if audio_array.ndim == 1:
#         audio = audio_array.float().unsqueeze(0)
#     else:
#         audio = audio_array.float()
#     # Resample if needed
#     if sr != target_sr:
#         audio = Resample(sr, target_sr)(audio)
#     if audio_norm_scale < 1.0:
#         audio = audio * audio_norm_scale
#     return audio

from vq_process import extract_vq_code_for_offline_training as Codec_model
def get_speech_token(input_waveform, input_features):
    
    """
    Extract speech token sequence using the encoder.
    It is assumed that encoder.encode_batch_feats returns a tensor whose shape could be (B, 1, seq_len) or (B, seq_len).
    If the returned shape is (B, 1, seq_len), squeeze out the 1st dimension.
    """
    with torch.no_grad():
        speech_tokens = Codec_model(
            input_waveform=input_waveform,
            input_features=input_features
        )
    if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
        speech_tokens = speech_tokens.squeeze(1)
    return speech_tokens 

# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

from torch.utils.data import DataLoader
sys.path.append('/mnt/fast/nobackup/users/jz01101/cy/LlasaEdit/LLaSA_training/finetune/online_finetune')

from tts_online_dataset_genshin_ata import WaveDataset, pad_audio_batch
from torch.utils.data import Subset

test_dataset = WaveDataset(
        test_data_split, 
        sampling_rate=16000, 
        tokenizer=tokenizer, 
        use_text=False, 
        task='ata', 
        text_guide=False, 
        mix_mode=False,
        mode='eval')
test_dataset = Subset(test_dataset, list(range(1000)))

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,         
    num_workers=0,        
    pin_memory=True,
    collate_fn=pad_audio_batch    
)
base_num = 128256 + 8
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for batch in tqdm(test_loader):
        padded_src_audios = batch["padded_src_audios"].cuda()
        padded_src_feats = batch["padded_src_feats"].cuda()
        src_audio_length_tensor = batch["src_audio_length_tensor"]
        padded_text_tokens = batch["padded_text_tokens"]
        text_length_tensor = batch["text_length_tensor"]
        src_name = batch['src_name']
        trg_name = batch['trg_name']

        batch_size = padded_src_audios.size(0)

        # Text tokens
        text_length_list = text_length_tensor.tolist()
        all_text_tokens = [
            padded_text_tokens[i, :text_length_list[i]].tolist()
            for i in range(batch_size)
        ]

        # Speech tokens
        src_audio_length_list = src_audio_length_tensor.tolist()
        src_speech_tokens_all = get_speech_token(
            input_waveform=padded_src_audios,
            input_features=padded_src_feats
        )  # (B, L)

        src_processed_speech_tokens = []
        for i in range(batch_size):
            src_tokens = src_speech_tokens_all[i, :src_audio_length_list[i]] + base_num
            src_tokens = [speech_understanding_start_id] + src_tokens.tolist() + [speech_understanding_end_id]
            src_processed_speech_tokens.append(src_tokens)

        # input_ids 构建
        max_total_length = 2048
        combined_tokens = []
        for text_tok, src_speech_tok in zip(all_text_tokens, src_processed_speech_tokens):
            # combined = text_tok + src_speech_tok
            combined = replace_tagged_token(text_tok, speech_understanding_start_id, src_speech_tok)
                
            # if len(combined) > max_total_length:
            #     continue
            # else:
            #     combined += [tokenizer.pad_token_id] * (max_total_length - len(combined))
            combined_tokens.append(combined)

        input_ids = torch.tensor(combined_tokens, dtype=torch.long).cuda()

        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=input_ids,
                max_length=2048,                  # 根据任务调小，避免过长胡言乱语
                eos_token_id=eos_token_id,
                do_sample=True,
                top_p=0.9,                       # 限制采样范围
                temperature=0.7,                 # 降低温度，减少发散
                repetition_penalty=1.2           # 惩罚重复片段（>1 抑制，<1 鼓励重复）
            )

        for i in range(batch_size):
            # extract generated speech token ids
            generated_ids = outputs[i][input_ids.shape[1]:-1]
            gen_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gen_speech_ids = extract_speech_ids(gen_tokens_str)

            gen_token_tensor = torch.tensor(gen_speech_ids).cuda().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                gen_wav = reconstruct_from_vq_code(gen_token_tensor)  # (1, 1, T)

            # 写入音频
            src = src_name[i]
            trg = trg_name[i]
            save_name = f"{save_path}/{src}_to_{trg}.wav"
            sf.write(save_name, gen_wav.squeeze(), 16000)
    

