import os
import sys
import jiwer
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from whisper.normalizers import EnglishTextNormalizer
from transformers import Wav2Vec2Processor, HubertForCTC, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq
sys.path.append('/mnt/fast/nobackup/users/yc01815/code/xcodec2')
from vq_process import extract_vq_code, reconstruct_from_vq_code


import random
random.seed(42)

asr_model_name="facebook/hubert-large-ls960-ft"

processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
asr_model = HubertForCTC.from_pretrained(asr_model_name, use_safetensors=True).cuda()
normalizer = EnglishTextNormalizer()


def transcribe(audio_input):
    import torch
    import torchaudio

    TARGET_SR = 16000
    wavs = []

    # 统一处理为列表
    if isinstance(audio_input, str):
        audio_list = [audio_input]
    elif isinstance(audio_input, list):
        audio_list = audio_input
    else:
        raise ValueError("audio_input must be a string or list of strings")

    for audio in audio_list:
        wav, sr = torchaudio.load(audio)
        if wav.shape[0] > 1: 
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != TARGET_SR: 
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR).to('cuda')
            wav = resampler(wav.cuda())
        wavs.append(wav.squeeze(0).cpu().numpy())

    input_values = processor(wavs, return_tensors="pt", sampling_rate=TARGET_SR, padding=True).input_values
    logits = asr_model(input_values.to('cuda')).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    transcription = [normalizer(trans) for trans in transcription]

    return transcription if len(transcription) > 1 else transcription[0]


def valid_name():
    audio_path = "/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/train_instruct.csv"
    df = pd.read_csv(audio_path)
    audio_full = df['audio_path'].tolist()
    base_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_train'
    batch = 5
    # __import__('ipdb').set_trace()

    for i in tqdm(range(0, len(audio_full), batch)):
        audio_list = audio_full[i:i+batch]
        audio_list = [f"{base_path}/{audio}" for audio in audio_list]
        try:
            trans_list = transcribe(audio_list)
        except:
            __import__('ipdb').set_trace()
        trans_gt = df['text'].tolist()[i:i+batch]
        trans_gt = [normalizer(text) for text in trans_gt]
        wer = jiwer.wer(trans_gt, trans_list)
        if wer > 0.2:
            print(trans_gt)
            print(trans_list)


def valid_vq():
    llasa_1b = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune/EVC_0708/checkpoint-28000'
    tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
    speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
    audio_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/bin_reprocess_gen/train_input_ids.memmap'
    shape_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/bin_reprocess_gen/train_input_ids_shape.npy'
    input_ids = np.memmap(audio_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))

    sampled = random.sample(range(70000), 1000)
    for i in tqdm(range(len(sampled))):
        input_seq = torch.tensor(input_ids[sampled[i]], dtype=torch.long)    
        under_end_pos = np.where(input_seq == speech_understanding_end_id)[0][0]
        gen_st_pos = np.where(input_seq == speech_generation_start_id)[0][0]
        # gen_end_pos = np.where(input_seq == speech_generation_end_id)[0][0]
        in_vq = input_seq[1:under_end_pos] - 128264
        out_vq = input_seq[gen_st_pos: ] - 128264
        in_recon_audio = reconstruct_from_vq_code(in_vq.unsqueeze(0).unsqueeze(0).cuda())
        out_recon_audio = reconstruct_from_vq_code(out_vq.unsqueeze(0).unsqueeze(0).cuda())
        sf.write('/mnt/fast/nobackup/users/yc01815/code/llasa/check_audio/source.wav', in_recon_audio, 16000)
        sf.write('/mnt/fast/nobackup/users/yc01815/code/llasa/check_audio/target.wav', out_recon_audio, 16000)
        transcribe_in = transcribe('/mnt/fast/nobackup/users/yc01815/code/llasa/check_audio/source.wav')
        transcribe_out = transcribe('/mnt/fast/nobackup/users/yc01815/code/llasa/check_audio/target.wav')
        # print(jiwer.wer(transcribe_in, transcribe_out))
        if jiwer.wer(transcribe_in, transcribe_out) > 0.5:
            print(jiwer.wer(transcribe_in, transcribe_out))
            print(transcribe_in)
            print(transcribe_out)


if __name__ == '__main__':
    valid_vq()