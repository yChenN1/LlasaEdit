 
import os
import glob
import torch
import random
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from whisper.normalizers import EnglishTextNormalizer


import random
random.seed(42) 

normalizer = EnglishTextNormalizer()


def init_worker(real_code_root_, gen_code_root_, meta_data_, tokenizer_, max_seq_len_, base_num_):
    global meta_data, real_code_root, gen_code_root, tokenizer, max_seq_len, base_num
    real_code_root = real_code_root_
    gen_code_root = gen_code_root_
    meta_data = meta_data_
    tokenizer = tokenizer_
    max_seq_len = max_seq_len_
    base_num = base_num_


def process_audio_id(audio_id):
    def get_code(code_root):
        code_file = code_root
        if not os.path.exists(code_file):
            print(f"{code_file} not exists")
            return None
        try:
            codes = np.load(code_file, allow_pickle=True)
        except Exception as e:
            print(f"load {code_file}: {e}")
            return None
        codes = codes.squeeze(0).squeeze(0)
        if codes.ndim != 1:
            print(f"The shape of {code_file} is wrong: {codes.ndim}")
            return None
    
        codes = base_num + torch.tensor(codes, dtype=torch.long)

        return codes
    src_root = f"{real_code_root}/{Path(audio_id['audio_path']).stem}.npy"
    trg_root = f"{gen_code_root}/{Path(audio_id['vc_path']).stem}.npy"
    src_code = get_code(src_root)
    trg_code = get_code(trg_root)
    transcript = normalizer(audio_id['text'])
    audio_path = src_root

    if src_code is None or trg_code is None:
        print(f"[SKIP] Missing code for {audio_path}")
        return (audio_path, None)
    
    '''
    input transcription
    text_with_special = f"<|TEXT_GENERATION_START|>{transcript}<|TEXT_GENERATION_END|>"
    text_with_special = f"<|TEXT_UNDERSTANDING_START|>{transcript}<|TEXT_UNDERSTANDING_END|>"
    encoded_text = tokenizer.encode_plus(
        text_with_special,
        add_special_tokens=False,
        return_tensors='np'
    )
    text_input_ids = encoded_text['input_ids'].squeeze(0)  # (text_len,)
    '''

    # input source audio
    text_with_special = f"<|TEXT_GENERATION_START|>{transcript}<|TEXT_GENERATION_END|>"
    encoded_text = tokenizer.encode_plus(
        text_with_special,
        add_special_tokens=False,
        return_tensors='np'
    )
    text_input_ids = encoded_text['input_ids'].squeeze(0)  # (text_len,)

    # input audio
    speech_under_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_under_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    audio_input_ids = np.array(
        [speech_under_start_id] +
        src_code.tolist() +
        [speech_under_end_id],
        dtype=np.int32
    )

    # output audio
    speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_gen_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    code_input_ids = np.array(
        [speech_gen_start_id] +
        trg_code.tolist() +
        [speech_gen_end_id],
        dtype=np.int32
    )
 
    # total_input_ids = np.concatenate([audio_input_ids, code_input_ids])
    # total_input_ids = np.concatenate([audio_input_ids, code_input_ids, text_input_ids])  # ESD_bin_reprocess_v2
    total_input_ids = np.concatenate([audio_input_ids, text_input_ids, code_input_ids])  # ESD_bin_reprocess_v3
 
    # if len(total_input_ids) > max_seq_len:
    #     total_input_ids = total_input_ids[:max_seq_len]

    if len(audio_input_ids) > int(max_seq_len * 2 / 7):
        print(f"[SKIP] Audio input too long: {len(audio_input_ids)} > 2/7 * {max_seq_len}")
        return (audio_path, None)

    if len(total_input_ids) < max_seq_len:
        padding_length = max_seq_len - len(total_input_ids)
        total_input_ids = np.pad(
            total_input_ids,
            (0, padding_length),
            'constant',
            constant_values=tokenizer.pad_token_id
        )
    else:
        total_input_ids = total_input_ids[:max_seq_len]  # truncate if needed

    return (audio_path, {'input_id': total_input_ids.astype(np.int32)})

    
def process_data(split, real_code_root, gen_code_root, meta_data, output_dir_tts, num_processes=4):
    os.makedirs(output_dir_tts, exist_ok=True)

    max_seq_len = 2048
 
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-1B-Instruct',
        model_max_length=2048,
        padding_side="right",
    )
 
    tokenizer.pad_token = tokenizer.eos_token
 
    special_tokens = [
        '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
        '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
        '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
        '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>'
    ]
    tokenizer.add_tokens(special_tokens)
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
 
    base_num = len(tokenizer)

    audio_ids = pd.read_csv(meta_data, header=0).to_dict(orient='records')
    random.shuffle(audio_ids)
 
    train_audio_ids = audio_ids
    df_shuffled = pd.DataFrame(train_audio_ids)

    num_processes = min(num_processes, multiprocessing.cpu_count())

 
    with multiprocessing.Pool(
        num_processes,
        initializer=init_worker,
        initargs=(real_code_root, gen_code_root, meta_data, tokenizer, max_seq_len, base_num)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_audio_id, train_audio_ids),
            total=len(train_audio_ids),
            desc="data processing"
        ))

    
    train_tts_input_ids_list = []
    skipped_audio_paths = set()

    for audio_path, res in results:
        if res is not None:
            train_tts_input_ids_list.append(res['input_id'])
        else:
            skipped_audio_paths.add(audio_path)
    skipped_audio_paths = [f"{Path(audio_path).stem}.wav" for audio_path in skipped_audio_paths]

    df_cleaned = df_shuffled[~df_shuffled['audio_path'].isin(skipped_audio_paths)]
    df_cleaned.to_csv(f"{output_dir_tts}/{split}_data.csv", index=False)
    
    init_worker(real_code_root, gen_code_root, meta_data, tokenizer, max_seq_len, base_num)

    train_tts_input_ids_array = np.array(train_tts_input_ids_list)

    train_tts_memmap_path = os.path.join(output_dir_tts, f'{split}_input_ids.memmap')
    train_tts_memmap = np.memmap(
        train_tts_memmap_path, dtype='int32', mode='w+', shape=train_tts_input_ids_array.shape
    )
    train_tts_memmap[:] = train_tts_input_ids_array[:]
    print(train_tts_memmap.shape, len(df_cleaned))
    del train_tts_memmap   
    np.save(os.path.join(output_dir_tts, f'{split}_input_ids_shape.npy'), train_tts_input_ids_array.shape)

    
    # Save combined C
    print(f"TTS memmap saved! {output_dir_tts}")

if __name__ == "__main__":
    
    split = 'valid'
    real_code_root = f'/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/vq_code_ear_expr_{split}'
    gen_code_root = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/vq_code_gen'

    os.makedirs(gen_code_root, exist_ok=True)
    meta_data = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/train_instruct_v2.csv'
    meta_data = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/valid_instruct_0_1000.csv'
    output_dir_tts = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/bin_reprocess_gen_transcription'
 
    num_processes = 8

    process_data(
        split,
        real_code_root,
        gen_code_root,
        meta_data,
        output_dir_tts,
        num_processes=num_processes
    )
