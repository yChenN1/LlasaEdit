 
import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from transformers import AutoTokenizer


def init_worker(code_root_, meta_data_, tokenizer_, max_seq_len_, base_num_):
    global meta_data, code_root, tokenizer, max_seq_len, base_num
    code_root = code_root_
    meta_data = meta_data_
    tokenizer = tokenizer_
    max_seq_len = max_seq_len_
    base_num = base_num_


def process_audio_id(audio_id):
    def get_code(code_root):
        code_file = code_root
        if not os.path.exists(code_file):
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
    src_root = f"{code_root}/{audio_id['src_path']}.npy"
    trg_root = f"{code_root}/{audio_id['trg_path']}.npy"
    src_code = get_code(src_root)
    trg_code = get_code(trg_root)
    instruct = audio_id['instruction']
    src_audio_root = f"{audio_id['src_path']}"
    trg_audio_root = f"{audio_id['trg_path']}"

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

    # input audio
    speech_under_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
    speech_under_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    audio_input_ids = np.array(
        [speech_under_start_id] +
        src_code.tolist() +
        [speech_under_end_id],
        dtype=np.int32
    )

    speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_gen_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    code_input_ids = np.array(
        [speech_gen_start_id] +
        trg_code.tolist() +
        [speech_gen_end_id],
        dtype=np.int32
    )
 
    total_input_ids = np.concatenate([audio_input_ids, code_input_ids])
 
    if len(total_input_ids) > max_seq_len:
        total_input_ids = total_input_ids[:max_seq_len]
    else:
        padding_length = max_seq_len - len(total_input_ids)
        total_input_ids = np.pad(
            total_input_ids,
            (0, padding_length),
            'constant',
            constant_values=tokenizer.pad_token_id
        )

    return {'input_id': total_input_ids.astype(np.int32), 'instruction': instruct, "audio_path": {'src_path': src_audio_root, 'trg_path': trg_audio_root}}

def process_data(code_root, meta_data, output_dir_tts, num_processes=4):
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
 
    val_audio_ids = audio_ids[-1000:]
    train_audio_ids = audio_ids[:-1000]

    num_processes = min(num_processes, multiprocessing.cpu_count())

 
    with multiprocessing.Pool(
        num_processes,
        initializer=init_worker,
        initargs=(code_root, meta_data, tokenizer, max_seq_len, base_num)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_audio_id, train_audio_ids),
            total=len(train_audio_ids),
            desc="data processing"
        ))
    train_tts_input_ids_list = [res['input_id'] for res in results if res is not None]
    train_instruct_list = [res['instruction'] for res in results if res is not None]
    train_audio_list = [{'source_path': res['audio_path']['src_path'], 'target_path': res['audio_path']['trg_path']} for res in results if res is not None]
    
    init_worker(code_root, meta_data, tokenizer, max_seq_len, base_num)
    val_tts_input_ids_list = []
    val_instruct_list = []
    val_audio_list = []
    for audio_id in tqdm(val_audio_ids, desc="valid data processing"):
        res = process_audio_id(audio_id)
        if res is not None:
            val_tts_input_ids_list.append(res['input_id'])
            val_instruct_list.append(res['instruction'])
            val_audio_list.append({'source_path': res['audio_path']['src_path'], 'target_path': res['audio_path']['trg_path']})
    
    if not (train_tts_input_ids_list or val_tts_input_ids_list):
        print("bug ")
        return

 
    all_ids = train_tts_input_ids_list + val_tts_input_ids_list
    max_total_token_len = max(len(ids) for ids in all_ids)
 
 
    os.makedirs(output_dir_tts, exist_ok=True)

 
    train_tts_input_ids_array = np.array(train_tts_input_ids_list)
    val_tts_input_ids_array = np.array(val_tts_input_ids_list)

    train_tts_memmap_path = os.path.join(output_dir_tts, 'train_input_ids.memmap')
    train_tts_memmap = np.memmap(
        train_tts_memmap_path, dtype='int32', mode='w+', shape=train_tts_input_ids_array.shape
    )
    train_tts_memmap[:] = train_tts_input_ids_array[:]
    del train_tts_memmap   

    val_tts_memmap_path = os.path.join(output_dir_tts, 'val_input_ids.memmap')
    val_tts_memmap = np.memmap(
        val_tts_memmap_path, dtype='int32', mode='w+', shape=val_tts_input_ids_array.shape
    )
    val_tts_memmap[:] = val_tts_input_ids_array[:]
    del val_tts_memmap   

    pd.DataFrame(train_instruct_list).to_csv(os.path.join(output_dir_tts, 'train_instruct.csv'), index=False)
    pd.DataFrame(val_instruct_list).to_csv(os.path.join(output_dir_tts, 'val_instruct.csv'), index=False)
    
    # Save combined CSV files
    train_combined = pd.DataFrame({
        'source': [audio['source_path'] for audio in train_audio_list],
        'target': [audio['target_path'] for audio in train_audio_list],
        'instruct': train_instruct_list
    })
    train_combined.to_csv(os.path.join(output_dir_tts, 'train_combined.csv'), index=False)

    val_combined = pd.DataFrame({
        'source': [audio['source_path'] for audio in val_audio_list],
        'target': [audio['target_path'] for audio in val_audio_list],
        'instruct': val_instruct_list
    })
    val_combined.to_csv(os.path.join(output_dir_tts, 'val_combined.csv'), index=False)


    np.save(os.path.join(output_dir_tts, 'train_input_ids_shape.npy'), train_tts_input_ids_array.shape)
    np.save(os.path.join(output_dir_tts, 'val_input_ids_shape.npy'), val_tts_input_ids_array.shape)

    print(f" TTS memmap  saved ! {output_dir_tts}")

if __name__ == "__main__":
 
    code_root = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/vq_code'
    meta_data = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/paired_emotion_with_instruct2.csv'
    output_dir_tts = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess'
 
    num_processes = 8

    process_data(
        code_root,
        meta_data,
        output_dir_tts,
        num_processes=num_processes
    )
