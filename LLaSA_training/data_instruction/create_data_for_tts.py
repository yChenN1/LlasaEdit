import pandas as pd
import torchaudio
import datasets
from tqdm import tqdm
import os
import math

tqdm.pandas()

def load_audio(path):
    try:
        array, sampling_rate = torchaudio.load(path)
        return {
            "path": path,
            "array": array.squeeze().numpy().astype("float32"),
            "sampling_rate": sampling_rate
        }
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def process_batch(df_batch, chunk_start_index, base_path, save_dir, chunk_size=2000):
    # 构造新结构
    new_rows = []
    for _, row in df_batch.iterrows():
        new_rows.append({
            "src_audio": f'{base_path}/{row["audio_path"]}',
            "trg_audio": f'{row["vc_path"]}',
            "text": row["text"],
            "src_caption": row["caption"],
            "trg_caption": row["caption_gen"],
            "src_instruct": row["src_instruct"],
            "trg_instruct": row["trg_instruct"],
        })

    new_df = pd.DataFrame(new_rows)

    # 加载音频为 array + sampling_rate 格式
    new_df["src_audio"] = new_df["src_audio"].progress_apply(load_audio)
    new_df["trg_audio"] = new_df["trg_audio"].progress_apply(load_audio)

    # 分块保存为多个 parquet 文件，每块最多 chunk_size 条
    num_chunks = math.ceil(len(new_df) / chunk_size)
    for j in range(num_chunks):
        chunk_df = new_df.iloc[j * chunk_size : (j + 1) * chunk_size]
        chunk_dataset = datasets.Dataset.from_pandas(chunk_df)
        chunk_idx = chunk_start_index + j
        chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:03d}.parquet")
        chunk_dataset.to_parquet(chunk_path)
        print(f"[INFO] Saved chunk {chunk_idx:03d} to {chunk_path}")
    
    return chunk_start_index + num_chunks

# ================= Main Logic =================
df = pd.read_csv("/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/valid_instruct.csv")
base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_valid'
save_dir = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/VST_chunks_valid"
os.makedirs(save_dir, exist_ok=True)

batch_size = 2000
total_chunks = 0

for start_idx in range(0, len(df), batch_size):
    end_idx = min(start_idx + batch_size, len(df))
    df_batch = df.iloc[start_idx:end_idx]
    total_chunks = process_batch(df_batch, total_chunks, base_path, save_dir)
