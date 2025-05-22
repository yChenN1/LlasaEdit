import os
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model
from datetime import datetime

# === Load Models ===
llasa_1b = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune_lora/EVC_0521/checkpoint-100000'
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
llm_model = AutoModelForCausalLM.from_pretrained(llasa_1b).eval().cuda()

codec_path = "HKUSTAudio/xcodec2"
codec_model = XCodec2Model.from_pretrained(codec_path).eval().cuda()


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


speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
eos_token_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

# === test xcodec2 ability ===
'''
audio_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0012/Angry/0012_000351.wav"
wav, sr = sf.read(audio_path)
assert sr == 16000, "Only supports 16kHz audio"

# === Encode Audio to Speech Tokens ===
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).cuda()  # Shape: (1, T)
with torch.no_grad():
    vq_code = codec_model.encode_code(input_waveform=wav_tensor)  # (1, 1, T_code)
speech_ids = vq_code[0, 0].cpu().numpy()
# speech_token_str = ids_to_tokens(speech_ids)

# === Save Reconstructed Original Audio ===
org_token_tensor = torch.tensor(speech_ids).cuda().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    org_wav = codec_model.decode_code(org_token_tensor)  # (1, 1, T)
sf.write("reconstructed_org.wav", org_wav[0, 0].cpu().numpy(), 16000)
'''

# === Input: eval audio set ===
audio_eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined.csv'
eval_list = pd.read_csv(audio_eval_path).to_dict(orient='records')
base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/'
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
elif "finetune" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/{today}/{step}"
else:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/other/{today}/{step}"

print("Save path:", save_path)
os.makedirs(save_path, exist_ok=True)

for audio in tqdm(eval_list):
    audio_path = f"{base_path}/{audio['source']}"
    instruct = audio['instruct']
    wav, sr = sf.read(audio_path)
    assert sr == 16000, "Only supports 16kHz audio"
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).cuda()  # Shape: (1, T)
    with torch.no_grad():
        vq_code = codec_model.encode_code(input_waveform=wav_tensor)  # (1, 1, T_code)
    speech_ids = vq_code[0, 0].cpu().numpy() + 128256 + 8

    # === Generate New Speech Tokens ===
    # formatted_input = "".join(speech_understanding_start_id) + "".join(speech_ids) + "".join(speech_understanding_end_id)

    formatted_input = torch.from_numpy(np.array(
            [speech_understanding_start_id] +
            speech_ids.tolist() +
            [speech_understanding_end_id],
            dtype=np.int32
        ))

    chat = [
                {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt', continue_final_message=True).to('cuda')
    input_ids = replace_tagged_token_torch(input_ids.squeeze(0), speech_understanding_start_id, formatted_input).unsqueeze(0)

    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=input_ids,
            max_length=2048,
            eos_token_id=eos_token_id,
            do_sample=True,
            top_p=1.0,
            temperature=0.8
        )

    generated_ids = outputs[0][input_ids.shape[1]:-1]
    gen_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    gen_speech_ids = extract_speech_ids(gen_tokens_str)

    # === Decode Generated Tokens ===
    gen_token_tensor = torch.tensor(gen_speech_ids).cuda().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        gen_wav = codec_model.decode_code(gen_token_tensor)  # (1, 1, T)
    sf.write(f"{save_path}/{Path(audio_path).stem}_{audio_path.split('/')[-2]}_to_{instruct.split(' ')[-1]}wav", gen_wav[0, 0].cpu().numpy(), 16000)


    

