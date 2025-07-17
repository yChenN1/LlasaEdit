from math import exp
import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('/mnt/fast/nobackup/users/yc01815/code/xcodec2')
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
llasa_1b = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune_lora/EVC_0710/checkpoint-11000'
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


def left_pad_sequences(sequences, pad_token_id):
    """
    sequences: List[torch.Tensor], each of shape (seq_len,)
    Returns: padded_tensor of shape (batch_size, max_len)
    """
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padded = torch.cat([torch.full((pad_len,), pad_token_id, dtype=seq.dtype), seq], dim=0)
        padded_seqs.append(padded)
    return torch.stack(padded_seqs, dim=0)



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
text_front = False

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
# audio_eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined.csv'
audio_eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess_v2/val_data.csv'
audio_eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/valid_instruct.csv'
eval_list = pd.read_csv(audio_eval_path).to_dict(orient='records')[:]
base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/'
base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_valid'
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
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/lora/{today}/{step}_2"
elif 'grpo' in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/grpo/{today}/{step}"
elif "finetune" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/{today}/{step}_train"
elif "text" in llasa_1b:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/text/{today}/{step}"
else:
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/other/{today}/{step}"

print("Save path:", save_path)
os.makedirs(save_path, exist_ok=True)


BATCH_SIZE = 1
eval_list = eval_list[:2000]  # Optional truncate

gen_text, gen_name, gt_text = [], [], []

for i in tqdm(range(0, len(eval_list), BATCH_SIZE), desc="Batch inference"):
    batch_items = eval_list[i:i+BATCH_SIZE]
    audio_paths, instructs, transcripts = [], [], []
        
    input_ids_batch = []
    speech_ids_list = []
    for audio in batch_items:
        audio_path = f"{base_path}/{audio['audio_path']}"
        instruct = audio['trg_instruct']
        transcript = audio['text']

        wav, sr = librosa.load(audio_path, sr=16000)
        assert sr == 16000, "Only supports 16kHz audio"

        with torch.no_grad():
            vq_code = extract_vq_code(wav)
        speech_ids = vq_code[0, 0].cpu().numpy() + 128256 + 8
        speech_ids_list.append(speech_ids)
        audio_paths.append(audio_path)
        instructs.append(instruct)
        transcripts.append(transcript)
    
        # === Prepare input_ids batch ===
        for instruct, speech_ids in zip(instructs, speech_ids_list):
            formatted_input = torch.tensor(
                [speech_understanding_start_id] + speech_ids.tolist() + [speech_understanding_end_id],
                dtype=torch.long
            )

        if use_text:
            system_content = (
                "You are an expert speech assistant. Your task is to generate an accurate and clear transcription of the input speech, and follow the given instruction to produce the appropriate speech."
                if text_front else
                "You are an expert speech assistant. Your task is to follow the given instruction to produce the appropriate speech, and generate an accurate and clear transcription of the input speech."
            )
            chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "The transcription of input speech is: <|TEXT_GENERATION_START|>" if text_front else "The converted speech is: <|SPEECH_GENERATION_START|>"}
            ]
        else:
            chat = [
                {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt', continue_final_message=True)[0]
        input_ids = replace_tagged_token_torch(input_ids, speech_understanding_start_id, formatted_input)
        input_ids_batch.append(input_ids)


    input_ids_batch = left_pad_sequences(input_ids_batch, tokenizer.pad_token_id).cuda()
    
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=input_ids_batch,
            max_length=2048,
            eos_token_id=eos_token_id,
            do_sample=True,
            top_p=1.0,
            temperature=0.95
        )

    # __import__('ipdb').set_trace()
    for j in range(len(batch_items)):
        gen_idx = (outputs[j] == speech_generation_start_id).nonzero(as_tuple=False)[0].item()
        end_idx = (outputs[j] == speech_generation_end_id).nonzero(as_tuple=False)[0].item()
        generated_ids = outputs[j][gen_idx + 1: end_idx]
        try:
            gen_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gen_speech_ids = extract_speech_ids(gen_tokens_str)
        except Exception as e:
            print(f"Error decoding batch idx {i+j}: {e}")
            continue

        # === Decode to waveform ===
        token_tensor = torch.tensor(gen_speech_ids).cuda().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            gen_wav = reconstruct_from_vq_code(token_tensor)

        out_path = f"{save_path}/{Path(audio_paths[j]).stem}.wav"
        sf.write(out_path, gen_wav, 16000)
        gen_name.append(out_path)
        gen_text.append("")  # 填空或插入text预估逻辑
        gt_text.append(transcripts[j])


df = pd.DataFrame({
"filename": gen_name,
"text_pre": gen_text,
'text_gt': gt_text
})

df.to_csv(f"{save_path}/output.csv", index=False)    

    

