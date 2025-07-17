 
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoFeatureExtractor
from torchaudio.transforms import Resample
from torch.utils.data import Dataset


# Adapt the dataset to the new format
class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, data, sampling_rate, tokenizer, audio_norm_scale: float = 1.0, root_dir: str = "", max_audio_duration: float = 41.0, use_instruction=False):
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
        self.use_instruction = use_instruction
    
    def __len__(self):
        # Each record corresponds to one sample
        return len(self.data)
    
    def __getitem__(self, index):
        max_retry = 10  # 避免死循环
        for _ in range(max_retry):
            item = self.data[index]
            transcription = item['text']
            # speaker = item['speaker']  # 'speaker' directly from the dataset
            src_audio_array = torch.tensor(item['src_audio']['array'])
            src_sr = item['src_audio']['sampling_rate']

            def process_audio(audio_array, sr, target_sr):
                if audio_array.ndim == 1:
                    audio = torch.tensor(audio_array, dtype=torch.float).unsqueeze(0)
                else:
                    audio = torch.tensor(audio_array, dtype=torch.float)
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
        
        trg_audio_array = torch.tensor(item['trg_audio']['array'])
        trg_sr = item['trg_audio']['sampling_rate']
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
        
        # Prepare the text for tokenization
        # text_with_special = f"<|TEXT_UNDERSTANDING_START|>{transcription}<|TEXT_UNDERSTANDING_END|>"
        if not self.use_instruction:
            style_instruction = item['trg_instruct']
            chat = [
                {"role": "user", "content": "{style_instruction}".format(style_instruction=style_instruction)},
                # {"role": "assistant", "content": f"Speaker {speaker}"}
            ]
        else:
            style_instruction = item['caption']
            chat = [
                {"role": "system", "content": "You are a helpful speech assistant. Your job is to generate speech that matches the provided style instruction as closely as possible."},
                {
                "role": "user",
                "content": (
                    "Convert the following text to speech based on the given style instruction.\n\n"
                    "Instruction: {instruction}\n\n"
                    "Text: {text}"
                ).format(instruction=style_instruction, text=text_with_special)
                }
            ]
        text_tokens = self.tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_length = text_tokens.size(0)
        
        # Return all the data
        return src_audio, src_feat, trg_audio, trg_feat, src_audio_length, trg_audio_length, text_tokens, text_length

def pad_audio_batch(batch):
    # audio_list, feat_list, audio_length_list, text_tokens_list, text_length_list = zip(*batch)
    src_audio_list, src_feat_list, trg_audio_list, trg_feat_list, src_audio_length_list, trg_audio_length_list, text_tokens_list, text_length_list = zip(*batch)
    
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
    }
    

    
 
    return {
        "padded_audios": padded_audios,
        "padded_feat_list": padded_feat_list,
        "audio_length": audio_length_tensor,
        "text_tokens": padded_text_tokens,
        "text_length": text_length_tensor,
        # "speakers": speaker_list,
    }

# # Example usage
# if __name__ == '__main__':
#     from datasets import load_dataset
#     from torch.utils.data import DataLoader
#     from collections import defaultdict
#     from tqdm import tqdm 
#     # Load the new dataset
#     dataset = load_dataset('simon3000/genshin-voice', cache_dir='/aifs4su/data/zheny/opensource/local_data160/genshin')
#     data_split = dataset['train']
    
#     # Initialize the tokenizer 
#     # tokenizer = AutoTokenizer.from_pretrained("HKUSTAudio/Llasa-1B")
    
#     # # Instantiate the dataset
#     # genshin_dataset = WaveDataset(data_split, sampling_rate=16000, tokenizer=tokenizer)
    
#     # # Create DataLoader
#     # loader = DataLoader(genshin_dataset, batch_size=2, collate_fn=pad_audio_batch)
    
#     # # Retrieve one batch
#     # batch = next(iter(loader))
#     # print("padded_audios shape:", batch["padded_audios"].shape)
#     # print("padded_feat_list shape:", batch["padded_feat_list"].shape)
#     # print("audio_length:", batch["audio_length"])
#     # print("text_tokens shape:", batch["text_tokens"].shape)
#     # print("text_length:", batch["text_length"])
#     # print("speakers:", batch["speakers"])
#     speaker_duration = defaultdict(float)
#     speaker_types = defaultdict(set)
#     speaker_languages = defaultdict(set)
 
#     for item in tqdm(data_split, desc="Processing items"):
#         speaker = item['speaker']
#         duration = item['audio']['array'].shape[0]/48000
#         speaker_type = item['type']
#         language = item['language']
        
#         speaker_duration[speaker] += duration
#         speaker_types[speaker].add(speaker_type)
#         speaker_languages[speaker].add(language)
 
if __name__ == '__main__':
    from datasets import load_dataset
    from collections import defaultdict
    from tqdm import tqdm 
    
    # Load the new dataset
    dataset = load_dataset('simon3000/genshin-voice', cache_dir='/aifs4su/data/zheny/opensource/local_data160/genshin')
    data_split = dataset['train']
    
 
    # Define your selected speakers (based on your previous list)
    selected_speakers = {
        'Paimon', 'Traveler', 'Nahida', 'Navia', 'Furina', 'Lyney', 'Layla', 'Neuvillette',
        'Kaveh', 'Tighnari', 'Alhaitham', 'Kaeya', 'Dehya', 'Zhongli', 'Cyno', 'Yoimiya',
        'Mualani', 'Ningguang', 'Nilou', 'Faruzan', 'Wriothesley', 'Collei', 'Thoma', 'Noelle',
        'Venti', 'Lynette', 'Charlotte', 'Diona', 'Yelan', 'Clorinde', 'Sigewinne', 'Beidou',
        'Gorou', 'Lisa', 'Yanfei', 'Xianyun', 'Chevreuse', 'Sucrose', 'Sayu', 'Ganyu', 'Chiori',
        'Chongyun', 'Freminet', 'Kachina', 'Barbara', 'Baizhu', 'Kirara', 'Emilie', 'Dainsleif',
        'Klee', 'Albedo', 'Dori', 'Eula', 'Xiao', 'Mona', 'Bennett', 'Amber', 'Xingqiu', 'Shenhe',
        'Childe', 'Kinich', 'Xiangling', 'Gaming', 'Jean', 'Diluc', 'Mavuika', 'Katheryne', 'Aeval',
        'Mika', 'Dunyarzad', 'Keqing', 'Candace'
    }

 

    # Filter the dataset based on selected speakers
    filtered_data_split = data_split.filter(lambda voice: voice['speaker'] in selected_speakers)

from datasets import load_dataset

if __name__ == '__main__':
    from tqdm import tqdm 
    
    # Load the new dataset
    dataset = load_dataset('simon3000/genshin-voice', cache_dir='/aifs4su/data/zheny/opensource/local_data160/genshin')
    data_split = dataset['train']
    
    # Define your selected speakers (based on your previous list)
    selected_speakers = { 
        'Paimon', 'Traveler', 'Nahida', 'Navia', 'Furina', 'Lyney', 'Layla', 'Neuvillette',
        'Kaveh', 'Tighnari', 'Alhaitham', 'Kaeya', 'Dehya', 'Zhongli', 'Cyno', 'Yoimiya',
        'Ningguang', 'Nilou', 'Faruzan', 'Wriothesley', 'Collei', 'Thoma', 'Noelle',
        'Venti', 'Lynette', 'Charlotte', 'Diona', 'Yelan', 'Clorinde', 'Sigewinne', 'Beidou',
        'Gorou', 'Lisa', 'Yanfei', 'Sucrose', 'Sayu', 'Ganyu', 'Chiori', 'Chongyun', 'Freminet',
        'Barbara', 'Baizhu', 'Kirara', 'Dainsleif', 'Klee', 'Albedo', 'Dori', 'Eula', 'Xiao',
        'Mona', 'Bennett', 'Amber', 'Xingqiu', 'Shenhe', 'Childe', 'Xiangling', 'Jean', 'Diluc',
        'Katheryne', 'Mika', 'Keqing', 'Candace'
    }


    # Filter the dataset based on selected speakers
    filtered_data_split = data_split.filter(lambda voice: voice['speaker'] in selected_speakers)

    # Save the filtered dataset to disk
    save_path = '/aifs4su/data/zheny/opensource/local_data160/genshin_filter'  # Specify your local path here
    filtered_data_split.save_to_disk(save_path)

    print(f"Dataset saved to {save_path}")
