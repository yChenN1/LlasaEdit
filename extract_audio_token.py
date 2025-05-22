import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig

from xcodec2.modeling_xcodec2 import XCodec2Model
 

model_path = "HKUSTAudio/xcodec2"  
 
model = XCodec2Model.from_pretrained(model_path)
model.eval().cuda()   

audio_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English'
audio_path = glob.glob(f'{audio_path}/**/*.wav', recursive=True)

vq_code_output_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset'  

# __import__('pdb').set_trace()
for audio_p in tqdm(audio_path):
    wav, sr = sf.read(audio_p)   
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)
    
    with torch.no_grad():
    # Only 16khz speech
    # Only supports single input. For batch inference, please refer to the link below.
        vq_code = model.encode_code(input_waveform=wav_tensor)
        save_path = f"{vq_code_output_path}/{audio_p.split('/')[6]}/{audio_p.split('/')[7]}/{audio_p.split('/')[8]}/{audio_p.split('/')[9]}"
        os.makedirs(save_path, exist_ok=True)
        
        # print("Code:", vq_code)  
        np.save(f"{save_path}/{audio_p.split('/')[10]}.npy", vq_code.cpu().numpy())
    # recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

 
# sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
# print("Done! Check reconstructed.wav")
