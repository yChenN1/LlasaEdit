# import os
# import glob
# import torch
# import soundfile as sf
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoConfig

# from xcodec2.modeling_xcodec2 import XCodec2Model
 

# model_path = "HKUSTAudio/xcodec2"  
 
# model = XCodec2Model.from_pretrained(model_path)
# model.eval().cuda()   

# audio_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English'
# audio_path = glob.glob(f'{audio_path}/**/*.wav', recursive=True)

# vq_code_output_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset'  

# # __import__('pdb').set_trace()
# for audio_p in tqdm(audio_path):
#     wav, sr = sf.read(audio_p)   
#     wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)
    
#     with torch.no_grad():
#     # Only 16khz speech
#     # Only supports single input. For batch inference, please refer to the link below.
#         vq_code = model.encode_code(input_waveform=wav_tensor)
#         save_path = f"{vq_code_output_path}/{audio_p.split('/')[6]}/{audio_p.split('/')[7]}/{audio_p.split('/')[8]}/{audio_p.split('/')[9]}"
#         os.makedirs(save_path, exist_ok=True)
        
#         # print("Code:", vq_code)  
#         np.save(f"{save_path}/{audio_p.split('/')[10]}.npy", vq_code.cpu().numpy())
#     # recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

 
# # sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
# # print("Done! Check reconstructed.wav")

import os
import glob
import torch
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoConfig

from xcodec2.modeling_xcodec2 import XCodec2Model


def save_vq_codes(audio_dir, save_dir, model):
    audio_files = glob.glob(f'{audio_dir}/**/*.wav', recursive=True)
    for audio_p in tqdm(audio_files, desc="Saving VQ codes to .npy"):
        wav, sr = sf.read(audio_p)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)

        with torch.no_grad():
            vq_code = model.encode_code(input_waveform=wav_tensor.cuda())

        save_path = os.path.join(
            save_dir,
            *audio_p.strip("/").split("/")[-5:-1]
        )
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"{os.path.splitext(os.path.basename(audio_p))[0]}.npy"),
                vq_code.cpu().numpy())
        

def reconstruct_and_save_wav(audio_list, save_dir, model):
    base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/'
    for audio_p in tqdm(audio_list, desc="Reconstructing and saving audio"):
        wav, sr = sf.read(f"{base_path}/{audio_p}")
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)

        with torch.no_grad():
            vq_code = model.encode_code(input_waveform=wav_tensor.cuda())
            recon_wav = model.decode_code(vq_code).cpu() 

        wav_out = recon_wav[0, 0, :].numpy()

        save_name = f"{Path(audio_p).stem}_{audio_p.split('/')[-2]}_to_{audio_p.split('/')[-2]}.wav"
        sf.write(f"{save_dir}/{save_name}", wav_out, samplerate=16000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['save_npy', 'reconstruct_wav'], required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="HKUSTAudio/xcodec2")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    model = XCodec2Model.from_pretrained(args.model_path)
    model.eval().cuda()

    data_list = pd.read_csv(args.input_path)['src_path'].to_list()


    if args.mode == "save_npy":
        save_vq_codes(data_list, args.output_path, model)
    elif args.mode == "reconstruct_wav":
        reconstruct_and_save_wav(data_list, args.output_path, model)


# python extract_audio_token.py \
#     --mode reconstruct_wav \
#     --input_path  /mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined_eval.csv \
#     --output_path /mnt/fast/nobackup/scratch4weeks/yc01815/llasa/reconstruction_codec