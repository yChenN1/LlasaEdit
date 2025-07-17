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
llasa_1b = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune/EVC_0712/checkpoint-64000'
# llasa_1b ='HKUSTAudio/Llasa-1B'
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
llm_model = AutoModelForCausalLM.from_pretrained(llasa_1b).eval().cuda()



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


use_text = False
text_front = False


import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

class SingleInputReader:
    def __init__(self, data_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.memmap_path = os.path.join(data_path, 'train_input_ids.memmap')
        self.shape_path = os.path.join(data_path, 'train_input_ids_shape.npy')
        self.instruct_path = os.path.join(data_path, 'train_data.csv')

        self.input_shape = tuple(np.load(self.shape_path))
        self.input_ids_memmap = np.memmap(self.memmap_path, dtype='int32', mode='r', shape=self.input_shape)
        self.instruct_list = pd.read_csv(self.instruct_path).to_dict(orient = 'records')

        # Tokens
        self.speech_generation_start_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.speech_understanding_start_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')

    def __len__(self):
        return self.input_shape[0]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids_memmap[idx], dtype=torch.long)
        instruct = self.instruct_list[idx]['trg_instruct']
        filename = f"{Path(self.instruct_list[idx]['audio_path']).stem}_to_{Path(self.instruct_list[idx]['vc_path']).stem}.wav"
        transcript = self.instruct_list[idx]['text']

        # 构造 chat
        chat = [
            {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)

        # 替换 token
        def replace_token(token_list, target_token, new_sequence):
            idx = token_list.index(target_token)
            return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

        speech_gen_positions = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0]
        if len(speech_gen_positions) == 0:
            return None
        speech_gen_idx = speech_gen_positions[0].item()

        try:
            speech_gen_end_idx = (input_ids == self.speech_generation_end_id).nonzero(as_tuple=True)[0].item()
        except:
            speech_gen_end_idx = 2048

        src_speech = input_ids[:speech_gen_idx]
        gen_speech = input_ids[speech_gen_idx: speech_gen_end_idx + 1]

        ids = replace_token(ids, self.speech_understanding_start_id, src_speech)
        # ids = replace_token(ids, self.speech_generation_start_id, gen_speech)

        return ids, filename, transcript

# ✅ 示例使用
if __name__ == "__main__":
    data_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/bin_reprocess_gen"
    tokenizer_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune/EVC_0712/checkpoint-86000"
    today = datetime.now().strftime("%Y%m%d")
    basename = os.path.basename(llasa_1b)
    if basename.startswith("checkpoint-"):
        step = basename.replace("checkpoint-", "")[:5]
    else:
        step = "unknown"
    save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/{today}/{step}_train_vq"
    os.makedirs(save_path, exist_ok=True)
    reader = SingleInputReader(data_path, tokenizer_path)

    speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
    text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
    eos_token_id = speech_generation_end_id

    __import__('ipdb').set_trace()
    for i in tqdm(range(len(reader))):
        input_ids, filename, transcript = reader[i]
        input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)
        if input_ids is not None:
            with torch.no_grad():
                outputs = llm_model.generate(
                    input_ids=input_ids,
                    max_length=2048,
                    eos_token_id=eos_token_id,
                    do_sample=True,
                    top_p=1.0,
                    temperature=0.95
                )
            generated_ids = outputs[0][input_ids.shape[1]:-1]    
            gen_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gen_speech_ids = extract_speech_ids(gen_tokens_str)
            gen_token_tensor = torch.tensor(gen_speech_ids).cuda().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                gen_wav = reconstruct_from_vq_code(gen_token_tensor)  # (1, 1, T)
            sf.write(f"{save_path}/{filename}.wav", gen_wav, 16000)
