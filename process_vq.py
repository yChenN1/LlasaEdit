import os
import sys
import glob
import librosa
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 添加 xcodec2 模块路径
sys.path.append('/mnt/fast/nobackup/users/yc01815/code/xcodec2')
from vq_process import extract_vq_code


def get_vq_from_csv(audio_gen_path, save_path, split, start_index=0, end_index=None):
    input_path = f"{audio_gen_path}/{split}_vc_spk_filtered.csv"
    df = pd.read_csv(input_path)

    if end_index is None or end_index > len(df):
        end_index = len(df)

    df = df.iloc[start_index:end_index].copy()
    os.makedirs(save_path, exist_ok=True)

    for audio in tqdm(df.to_dict(orient='records'), desc=f"[{split}] Extracting VQ from {start_index} to {end_index}"):
        audio_path = audio['vc_path']
        audio_name = Path(audio_path).stem
        save_file = f"{save_path}/{audio_name}.npy"

        try:
            wav, _ = librosa.load(audio_path, sr=16000)
            vq = extract_vq_code(wav).cpu().numpy()
            np.save(save_file, vq)
        except Exception as e:
            print(f"[ERROR] Failed to extract VQ for {audio_path}: {e}")


def get_vq_from_folder(audio_path, save_path, start_index=0, end_index=None):
    audio_files = sorted(glob.glob(f'{audio_path}/**/*.wav', recursive=True))

    if end_index is None or end_index > len(audio_files):
        end_index = len(audio_files)

    os.makedirs(save_path, exist_ok=True)
    subset_files = audio_files[start_index:end_index]

    for audio_file in tqdm(subset_files, desc=f"[Real] Extracting VQ from {start_index} to {end_index}"):
        audio_name = Path(audio_file).stem
        save_file = f"{save_path}/{audio_name}.npy"

        if os.path.exists(save_file):
            continue
        try:
            wav, _ = librosa.load(audio_file, sr=16000)
            vq = extract_vq_code(wav).cpu().numpy()
            np.save(save_file, vq)

        except Exception as e:
            print(f"[ERROR] Failed to extract VQ for {audio_file}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default="train", help="Dataset split")
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--target", type=str, choices=["csv", "real", "both"], default="csv", help="Which VQ extraction to run")

    args = parser.parse_args()

    base_gen_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1"
    save_gen_vq = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/vq_code_gen"

    base_real_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_valid"
    save_real_vq = "/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/vq_code_ear_expr_valid"

    if args.target == "csv":
        get_vq_from_csv(base_gen_path, save_gen_vq, args.split, args.start_index, args.end_index)
    elif args.target == "real":
        get_vq_from_folder(base_real_path, save_real_vq, args.start_index, args.end_index)
    elif args.target == "both":
        get_vq_from_csv(base_gen_path, save_gen_vq, args.split, args.start_index, args.end_index)
        get_vq_from_folder(base_real_path, save_real_vq, args.start_index, args.end_index)


if __name__ == "__main__":
    main()
