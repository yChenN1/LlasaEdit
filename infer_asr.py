import os
import sys
import glob
import torch
import jiwer
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from datetime import datetime
from torchaudio.transforms import Resample
from whisper.normalizers import EnglishTextNormalizer
sys.path.append('/mnt/fast/nobackup/users/jz01101/cy/LlasaEdit/xcodec2')
from vq_process import extract_vq_code, reconstruct_from_vq_code
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer

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


llasa_1b ='/mnt/fast/nobackup/scratch4weeks/jz01101/llasa/finetune/0730_asr_asr_llasa1b_5e-4/checkpoint-50000'

# llasa_1b = "HKUSTAudio/Llasa-1B"
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
model.eval() 
model.to('cuda')


def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


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
speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
base_path = f'/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_valid'


def get_device():
    """获取可用设备"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_id: str, device: str):
    """加载语音识别模型"""
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return asr_pipeline, processor

device = get_device()
model_id = "openai/whisper-large-v3"
asr_pipeline, processor = load_model(model_id, device)
normalizer = EnglishTextNormalizer()


def process_audio(audio_array, sr, target_sr=16000):
    if audio_array.ndim == 1:
        audio = torch.tensor(audio_array, dtype=torch.float).unsqueeze(0)
    else:
        audio = torch.tensor(audio_array, dtype=torch.float)
    # Resample if needed
    if sr != target_sr:
        audio = Resample(sr, target_sr)(audio)
    return audio
    

#TTS start!
def gen_audio(data, save_path, mode='tts'):
    for item in tqdm(data):
        with torch.no_grad():
            input_text = item['text']
            src_audio_array = torch.tensor(item['src_audio']['array'])
            sr = item['src_audio']['sampling_rate']
            transcription = item['text']
            wav = process_audio(src_audio_array, sr)

            with torch.no_grad():
                vq_code = extract_vq_code(wav.numpy().squeeze(0))  # (1, 1, T_code)
                speech_ids = vq_code[0, 0].cpu().numpy() + 128256 + 8

            formatted_input = torch.from_numpy(np.array(
            [speech_understanding_start_id] +
            speech_ids.tolist() +
            [speech_understanding_end_id],
            dtype=np.int32
        ))

            # Tokenize the text
            # text_with_special = f"<|TEXT_GENERATION_START|>{transcription}<|TEXT_GENERATION_END|>"
            chat = [
                {"role": "user", "content": 'Convert speech to text.' + "<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "<|TEXT_GENERATION_START|>"}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = replace_tagged_token_torch(input_ids.squeeze(0), speech_understanding_start_id, formatted_input).unsqueeze(0)
            input_ids = input_ids.to('cuda')
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            text_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')

            # Generate the speech autoregressively
            outputs = model.generate(
                input_ids,
                max_length=2048,  # We trained our model with a max length of 2048
                eos_token_id=text_end_id,
                do_sample=True,    
                top_p=0.95,           #  Adjusts the diversity of generated content
                temperature=0.95,   #  Controls randomness in output
                repetition_penalty=1.2
            )
            # Extract the speech tokens
            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gen_text = ''.join(speech_tokens)
            __import__('ipdb').set_trace()   
        
        print(gen_text + '-----' + input_text)

import torchaudio
import jiwer

def check_data(data):
     for item in tqdm(data):
        with torch.no_grad():
            src_audio_array = torch.tensor(item['src_audio']['array'])
            sr = item['src_audio']['sampling_rate']
            gt_transcription = normalizer(item['text'])
            # 假设音频是 1D，需要转为 (1, T) 形状
            if src_audio_array.ndim == 1:
                src_audio_array = src_audio_array.unsqueeze(0)  # [1, T]
            torchaudio.save("output.wav", src_audio_array, sr)
            transcription = asr_pipeline("output.wav")['text'] # gt
            transcription = normalizer(transcription)
            if jiwer.wer(gt_transcription, transcription) > 0.5:
                print(gt_transcription, transcription)


if __name__ == '__main__':
    train_data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks/chunk_000.parquet'
    eval_data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks_valid/chunk_valid.parquet'

    
    from datasets import load_dataset
    train_data = load_dataset(
            'parquet',
            data_files={
                'train': [
                    train_data_path,
                ]
            },
            split='train',
        )
    eval_data = load_dataset(
            'parquet',
            data_files={
                'train': [
                    eval_data_path,
                ]
            },
            split='train',
        )
    save_path = './'
    # check_data(train_data)
    gen_audio(train_data, save_path)