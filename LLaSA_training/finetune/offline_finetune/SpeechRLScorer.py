import os
import sys
import math
import uuid
import glob
import torch
import shutil
import torchaudio
from jiwer import wer
import soundfile as sf
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from xcodec2.modeling_xcodec2 import XCodec2Model
from whisper.normalizers import EnglishTextNormalizer
sys.path.append('/mnt/fast/nobackup/users/yc01815/code/xcodec2')
from vq_process import load_models, extract_vq_code, reconstruct_from_vq_code
from transformers import Wav2Vec2Processor, HubertForCTC, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq


class SpeechRLScorer:
    def __init__(self, asr_model_name="facebook/hubert-large-ls960-ft", vocoder_name='HKUSTAudio/xcodec2', llasa_1b='/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/finetune/EVC_0521/checkpoint-20000'):
        self.processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.asr_model = HubertForCTC.from_pretrained(asr_model_name, use_safetensors=True).cuda()
        # self.codec_model = XCodec2Model.from_pretrained(vocoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
        self.normalizer = EnglishTextNormalizer()
        # self.codec_model.eval()
        self.asr_model.eval()


    def extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids
    
    def token2wav(self, audio_token, gen_length, path):
        '''gen_length: prompt 的 token 数量（用于切 response）'''
        file_name = []
        # speech_tokens_list = []
        # for i in range(len(audio_token)):
        #     speech_tokens_list.append(self.extract_speech_ids(self.tokenizer.batch_decode(audio_token[i], skip_special_tokens=True)))
        # try: 
        #     speech_tokens = torch.from_numpy((np.array(speech_tokens_list))).cuda().unsqueeze(1)        
        # except:
        #     __import__('ipdb').set_trace()
        # gen_wav = reconstruct_from_vq_code(speech_tokens)
        # for i in range(len(gen_wav)):
        #     sf.write(f"{path}/{i}.wav", gen_wav[i, :], 16000)
        #     file_name.append(f"{path}/{i}.wav")
        # __import__('pdb').set_trace()
        for i in range(len(audio_token)):
            speech_tokens = self.tokenizer.batch_decode(audio_token[i], skip_special_tokens=True)
            end_index = speech_tokens.index('<|SPEECH_GENERATION_END|>') if '<|SPEECH_GENERATION_END|>' in speech_tokens else len(speech_tokens)
            trimmed_tokens = speech_tokens[:end_index]
            speech_tokens = self.extract_speech_ids(trimmed_tokens)
            gen_wav = reconstruct_from_vq_code(torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0))
            sf.write(f"{path}/{i}.wav", gen_wav, 16000)
            file_name.append(f"{path}/{i}.wav")
    
        return file_name
       

    def transcribe(self, audio_list):
        TARGET_SR = 16000
        wavs = []
        for audio in audio_list:
            wav, sr = torchaudio.load(audio)
            # wav = wav.to("cuda")
            if wav.shape[0] > 1: 
                wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != TARGET_SR: 
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR).to('cuda')
                wav = resampler(wav)
            wavs.append(wav.squeeze(0))

        # __import__('pdb').set_trace()
        wavs = [w.cpu().numpy() for w in wavs]
        input_values = self.processor(wavs, return_tensors="pt", sampling_rate=TARGET_SR, padding=True).input_values
        logits =  self.asr_model(input_values.to('cuda').squeeze(0)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        transcription = [self.normalizer(trans) for trans in transcription]

        return transcription

    def compute_reward(self, prompts, completions, completion_ids, **kwargs):
        uid_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/temp/{str(uuid.uuid4())}"
        os.makedirs(uid_path, exist_ok=False)
        audio_list = self.token2wav(completions, len(completion_ids), uid_path)
        pred = self.transcribe(audio_list)
        text_gt = [self.normalizer(text) for text in kwargs['expected_text']]
        reward = [math.exp(-wer(text_gt[i], pred[i])) for i in range(len(text_gt))]
    
        if os.path.isdir(uid_path):
            shutil.rmtree(uid_path)
        return reward, pred
    

if __name__ == '__main__':
    model =SpeechRLScorer()
    completions = torch.load('/mnt/fast/nobackup/users/yc01815/code/llasa/LLaSA_training/finetune/offline_finetune/abc.pt')
    completion_ids = torch.load('/mnt/fast/nobackup/users/yc01815/code/llasa/LLaSA_training/finetune/offline_finetune/bcd.pt')
    prompt = ['Hello world', 'Hello world', 'Hello world', 'Hello world']
    model.compute_reward(prompt, completions, completion_ids)

