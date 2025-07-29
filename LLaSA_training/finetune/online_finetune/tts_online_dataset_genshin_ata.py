 
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoFeatureExtractor
from torchaudio.transforms import Resample
from torch.utils.data import Dataset

torch.set_printoptions(profile='full')


# Adapt the dataset to the new format
class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, data, sampling_rate, tokenizer, audio_norm_scale: float = 1.0, root_dir: str = "", max_audio_duration: float = 41.0, use_text=False, task='ata',text_guide=False, mix_mode=False):
        """
        data: A list of data entries, each containing 'audio', 'transcription', 'speaker', etc.
        tokenizer: A tokenizer used to convert text into tokens.
        max_audio_duration: Maximum audio duration in seconds (default: 41 seconds).
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.root_dir = root_dir
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.tokenizer = tokenizer
        self.max_audio_frames = int(max_audio_duration * self.sampling_rate)  # Maximum number of frames for the given max duration
        self.use_text = use_text  # wheather use text transcription as input or output
        self.task = task  # asr or audio to audio
        self.text_guide = text_guide # base use_text, whether use text as input
        self.mix_mode = mix_mode

        # Prepare the text for tokenization
        if self.mix_mode:
            p = random.random()
            self.task == 'ata' if p > 0.3 else 'tts'
    
    def __len__(self):
        # Each record corresponds to one sample
        return len(self.data) * 2
    
    def __getitem__(self, index):
        max_retry = 10  # 避免死循环
        base_index = index // 2
        mirror = index % 2 == 1 

        for _ in range(max_retry):
            item = self.data[base_index]
            if not mirror:
                src_audio_array = torch.tensor(item['src_audio']['array'])
                src_sr = item['src_audio']['sampling_rate']
                style_instruction = item['trg_instruct']
                trg_audio_array = torch.tensor(item['trg_audio']['array'])
                trg_sr = item['trg_audio']['sampling_rate']
            else:
                src_audio_array = torch.tensor(item['trg_audio']['array'])
                src_sr = item['trg_audio']['sampling_rate']
                style_instruction = item['src_instruct']
                trg_audio_array = torch.tensor(item['src_audio']['array'])
                trg_sr = item['src_audio']['sampling_rate']

            def process_audio(audio_array, sr, target_sr):
                if audio_array.ndim == 1:
                    audio = audio_array.float().unsqueeze(0)
                else:
                    audio = audio_array.float()
                # Resample if needed
                if sr != target_sr:
                    audio = Resample(sr, target_sr)(audio)
                if self.audio_norm_scale < 1.0:
                    audio = audio * self.audio_norm_scale
                return audio

            src_audio = process_audio(src_audio_array, src_sr, self.sampling_rate)
            src_audio_length_in_frames = src_audio.shape[1]
            if src_audio_length_in_frames < self.max_audio_frames / 2:
                break
            else:
                 index = random.randint(0, len(self.data) - 1)
        
        trg_audio = process_audio(trg_audio_array, trg_sr, self.sampling_rate)

        # Trim or pad audio to the max duration
        trg_audio_length_in_frames = trg_audio.shape[1]

        if src_audio_length_in_frames + trg_audio_length_in_frames > self.max_audio_frames:
            trg_audio = trg_audio[:, :self.max_audio_frames - src_audio_length_in_frames]  # Trim audio to the max allowed length
        # elif audio_length_in_frames < self.max_audio_frames:
        #     padding = self.max_audio_frames - audio_length_in_frames
        #     audio = F.pad(audio, (0, padding), mode='constant', value=0)  # Pad audio to the max allowed length
        
        # Pad 160 samples on both ends
        src_audio_pad = F.pad(src_audio, (160, 160))
        trg_audio_pad = F.pad(trg_audio, (160, 160))
        
        # Extract features
        src_feat = self.feature_extractor(
            src_audio_pad,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).data["input_features"]

        trg_feat = self.feature_extractor(
            trg_audio_pad,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).data["input_features"]
        
        # Calculate audio length in frames
        src_audio_length = int(src_audio.shape[1] / self.hop_length)
        trg_audio_length = int(trg_audio.shape[1] / self.hop_length)
        
        style_instruction = item['trg_instruct']
        

        if self.task == 'tts':
            transcription = item['text']
            text_with_special = f"<|TEXT_UNDERSTANDING_START|>{transcription}<|TEXT_UNDERSTANDING_END|>"
            chat = [
                {"role": "user", "content": 'Convert text to speech.' + text_with_special}
            ]

        elif self.task == 'asr':
            transcription = item['text']
            text_with_special = f"<|TEXT_GENERATION_START|>{transcription}<|TEXT_GENERATION_END|>"
            chat = [
                {"role": "user", "content": 'Convert speech to text.' + text_with_special}
            ]

        else:   
            if not self.use_text:
                chat = [
                    {"role": "user", "content": "{style_instruction}".format(style_instruction=style_instruction)},
                    # {"role": "assistant", "content": f"Speaker {speaker}"}
                ]
            else:
                transcription = item['text']
                if self.text_guide:
                    text_with_special = f"<|TEXT_UNDERSTANDING_START|>{transcription}<|TEXT_UNDERSTANDING_END|>"
                else:   
                    text_with_special = f"<|TEXT_GENERATION_START|>{transcription}<|TEXT_GENERATION_END|>"

                chat = [
                    {"role": "system", "content": "You are an expert speech assistant. Your task is to generate an accurate transcription of the input speech, and follow the given instruction to convert speech that matches the provided style instruction as closely as possible."},
                    {
                    "role": "user",
                    "content": (
                        "Instruction: {instruction}\n\n"
                        "Text: {text}"
                    ).format(instruction=style_instruction, text=text_with_special)
                    }
                ]
        text_tokens = self.tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_length = text_tokens.size(0)
        
        # Return all the data
        return src_audio, src_feat, trg_audio, trg_feat, src_audio_length, trg_audio_length, text_tokens, text_length, self.task

def pad_audio_batch(batch):
    # audio_list, feat_list, audio_length_list, text_tokens_list, text_length_list = zip(*batch)
    src_audio_list, src_feat_list, trg_audio_list, trg_feat_list, src_audio_length_list, trg_audio_length_list, text_tokens_list, text_length_list, task = zip(*batch)
    
    # Pad source audio
    max_src_length_feat = max(feat.shape[1] for feat in src_feat_list)
    max_src_audio_length = max_src_length_feat * 320
    padded_src_audios = []
    for audio in src_audio_list:
        padding = max_src_audio_length - audio.shape[1]
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode='constant', value=0)
        else:
            padded_audio = audio[:, :max_src_audio_length]
        padded_src_audios.append(padded_audio)
    padded_src_audios = torch.stack(padded_src_audios)

    # Pad target audio
    max_trg_length_feat = max(feat.shape[1] for feat in trg_feat_list)
    max_trg_audio_length = max_trg_length_feat * 320
    padded_trg_audios = []
    for audio in trg_audio_list:
        padding = max_trg_audio_length - audio.shape[1]
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode='constant', value=0)
        else:
            padded_audio = audio[:, :max_trg_audio_length]
        padded_trg_audios.append(padded_audio)
    padded_trg_audios = torch.stack(padded_trg_audios)

    # Pad source features
    padded_src_feats = []
    for feat in src_feat_list:
        padding = max_src_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode='constant', value=0)
        padded_src_feats.append(padded_feat)
    padded_src_feats = torch.stack(padded_src_feats)

    # Pad target features
    padded_trg_feats = []
    for feat in trg_feat_list:
        padding = max_trg_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode='constant', value=0)
        padded_trg_feats.append(padded_feat)
    padded_trg_feats = torch.stack(padded_trg_feats)

    # Convert lengths to tensors
    src_audio_length_tensor = torch.tensor(src_audio_length_list, dtype=torch.long)
    trg_audio_length_tensor = torch.tensor(trg_audio_length_list, dtype=torch.long)

    # Pad text tokens
    max_text_len = max(t.size(0) for t in text_tokens_list)
    padded_text_tokens = []
    for t in text_tokens_list:
        pad_len = max_text_len - t.size(0)
        if pad_len > 0:
            t_padded = F.pad(t, (0, pad_len), value=0)
        else:
            t_padded = t
        padded_text_tokens.append(t_padded)
    padded_text_tokens = torch.stack(padded_text_tokens)
    
    # Convert lengths to tensors
    src_audio_length_tensor = torch.tensor(src_audio_length_list, dtype=torch.long)
    trg_audio_length_tensor = torch.tensor(trg_audio_length_list, dtype=torch.long)
    text_length_tensor = torch.tensor(text_length_list, dtype=torch.long)

    return {
        "padded_src_audios": padded_src_audios,
        "padded_src_feats": padded_src_feats,
        "src_audio_length_tensor": src_audio_length_tensor,
        "padded_trg_audios": padded_trg_audios,
        "padded_trg_feats": padded_trg_feats,
        "trg_audio_length_tensor": trg_audio_length_tensor,
        "padded_text_tokens": padded_text_tokens,
        "text_length_tensor": text_length_tensor,
        "task": task[0]
    }