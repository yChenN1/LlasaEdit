import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
from pathlib import Path
from transformers import Wav2Vec2Processor, HubertForCTC, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq


val_data_root = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/bin_reprocess_gen/train_data.csv'
val_df = pd.read_csv(val_data_root)
val_audio_list = val_df['src_path'].tolist()
val_audio_trans = val_df['transcription'].tolist()


asr_model_name="facebook/hubert-large-ls960-ft"
processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
asr_model = HubertForCTC.from_pretrained(asr_model_name, use_safetensors=True).cuda()
normalizer = EnglishTextNormalizer()
asr_model.eval()


def transcribe(audio):
        TARGET_SR = 16000
        wav, sr = torchaudio.load(audio)
        wav = wav.to("cuda")
        if wav.shape[0] > 1: 
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != TARGET_SR: 
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR).to('cuda')
            wav = resampler(wav)

        input_values = processor(wav, return_tensors="pt", sampling_rate=TARGET_SR, padding=True).input_values
        logits =  asr_model(input_values.to('cuda').squeeze(0)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        # transcription = [normalizer(trans) for trans in transcription]

        return transcription[0]

import pandas as pd
wrong_file = []
for audio, audio_trans in tqdm(zip(val_audio_list, val_audio_trans)):
    audio_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/{audio}"
    pred = transcribe(audio_path)
    score = wer(normalizer(audio_trans), normalizer(pred))
    # print(audio + '------' + normalizer(audio_trans) + '------' + normalizer(pred))
    wrong_file.append([audio, audio_trans, pred, score])
    
pd.DataFrame(wrong_file, columns=['path', 'gt', 'pred', 'score']).to_csv('wrong_file.csv', index=False)