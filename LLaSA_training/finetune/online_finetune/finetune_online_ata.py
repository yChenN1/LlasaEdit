import os
import sys
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    PreTrainedModel
)
import transformers
import wandb
from itertools import islice
from datasets import load_from_disk
def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

from tts_online_dataset_genshin_ata import WaveDataset, pad_audio_batch
sys.path.append('/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/xcodec2')
from vq_process import extract_vq_code_for_offline_training as Codec_model
from vq_process import reconstruct_from_vq_code
from peft import LoraConfig, get_peft_model

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class llm_with_codec_model(PreTrainedModel):
    def __init__(self, config, llm: nn.Module, encoder: Callable, tokenizer=None, use_text=False, task='ata', text_guide=False):
        """
        Parameters:
          - config: Model configuration object (should contain or specify the llm's name/path)
          - llm: Causal language model (e.g., AutoModelForCausalLM) used to predict speech tokens
          - encoder: Speech codec model (must implement encode_batch_feats(input_waveform, input_features))
          - tokenizer: Tokenizer
        """
        super().__init__(config)
        self.config = config
        self.llm = llm
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.base_num = 128256 + 8  # length of llama tokenizer + 8 new special tokens
        # Get special token ids for speech generation
        self.speech_gen_start_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_gen_end_id   = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.speech_understanding_start_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id   = self.tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
        self.text_gen_start_id = self.tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_gen_end_id   = self.tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        
        self.use_text = use_text
        self.task = task
        self.text_guide = text_guide


    def get_speech_token(self, input_waveform, input_features):
        """
        Extract speech token sequence using the encoder.
        It is assumed that encoder.encode_batch_feats returns a tensor whose shape could be (B, 1, seq_len) or (B, seq_len).
        If the returned shape is (B, 1, seq_len), squeeze out the 1st dimension.
        """
        with torch.no_grad():
            speech_tokens = self.encoder(
                input_waveform=input_waveform,
                input_features=input_features
            )
        if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
            speech_tokens = speech_tokens.squeeze(1)
        return speech_tokens 

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, **batch):
        """
        Forward pass implementing the TTS training procedure:
          1. The dataset returns a tokenized text prompt (key "text_tokens") along with its actual length ("text_length").
          2. Use the encoder to extract speech tokens, then truncate based on "audio_length", and add 
             <|SPEECH_GENERATION_START|> and <|SPEECH_GENERATION_END|> tokens at the beginning and end.
          3. For each sample, concatenate the text tokens (only the valid portion) with the processed speech tokens.
             If the total length is less than 2048, pad; otherwise, truncate to 2048.
          4. Construct labels: set the text portion (the first text_len tokens) to ignore_index so that only the speech token part contributes to the loss.
        """
        # Retrieve tensors from the batch
        padded_src_audios = batch["padded_src_audios"]                  # Tensor, shape (B, 1, T)
        padded_src_feats = batch["padded_src_feats"]                    # Tensor, shape (B, 1, frames, feat_dim)
        src_audio_length_tensor = batch["src_audio_length_tensor"]      # Tensor, shape (B,)
        padded_trg_audios = batch["padded_trg_audios"]                  # Tensor,  shape (B, 1, T)
        padded_trg_feats = batch["padded_trg_feats"]                    # Tensor, shape (B, 1, frames, feat_dim)
        trg_audio_length_tensor = batch["trg_audio_length_tensor"]      # Tensor, shape (B,)
        padded_text_tokens = batch["padded_text_tokens"]                # Tensor, shape (B, L_text_padded)
        text_length_tensor = batch["text_length_tensor"]                # Tensor, shape (B,)
        
        batch_size = padded_src_audios.size(0)
    
        # For the text prompt, extract the actual token list for each sample
        text_length_list = text_length_tensor.tolist()
        all_text_tokens = []
        for i in range(batch_size):
            tokens = padded_text_tokens[i, :text_length_list[i]].tolist()
            all_text_tokens.append(tokens)
        
        # Use the encoder to extract the speech token sequence
        src_speech_tokens_all = self.get_speech_token(
            input_waveform=padded_src_audios,
            input_features=padded_src_feats
        )  # Expected shape: (B, seq_len)
        src_audio_length_list = src_audio_length_tensor.tolist()

        if self.task == "asr":
            trg_speech_tokens_all = None
            trg_audio_length_list = [0] * batch_size
        else:
            trg_speech_tokens_all = self.get_speech_token(
                input_waveform=padded_trg_audios,
                input_features=padded_trg_feats
            )  # Expected shape: (B, seq_len)
            trg_audio_length_list = trg_audio_length_tensor.tolist()

        trg_processed_speech_tokens = []
        src_processed_speech_tokens = []
        
        for i in range(batch_size):
            src_valid_length = src_audio_length_list[i]
            # Extract the valid part of the speech tokens and convert to list
            src_tokens = src_speech_tokens_all[i, :src_valid_length] + self.base_num
            src_tokens = src_tokens.tolist()
            # Add special tokens at the beginning and end
            src_tokens = [self.speech_understanding_start_id] + src_tokens + [self.speech_understanding_end_id]
            src_processed_speech_tokens.append(src_tokens)

            if self.task == "asr":
                trg_tokens = []
            else:
                trg_valid_length = trg_audio_length_list[i]
                # Extract the valid part of the speech tokens and convert to list
                trg_tokens = trg_speech_tokens_all[i, :trg_valid_length] + self.base_num
                trg_tokens = trg_tokens.tolist()
                # Add special tokens at the beginning and end
                trg_tokens = [self.speech_gen_start_id] + trg_tokens + [self.speech_gen_end_id]
            trg_processed_speech_tokens.append(trg_tokens)

        # Concatenate the text tokens and the processed speech tokens for each sample, ensuring a fixed length of 2048
        combined_tokens = []
        max_total_length = 2048
        for text_tok, src_speech_tok, trg_speech_tok in zip(all_text_tokens, src_processed_speech_tokens, trg_processed_speech_tokens):
            if self.task == 'asr' or (self.use_text and not self.text_guide):
                text_gen_pos = text_tok.index(self.text_gen_start_id)
                combined = text_tok[:text_gen_pos] + src_speech_tok + text_tok[text_gen_pos:] + trg_speech_tok
            else:
                combined = text_tok + src_speech_tok + trg_speech_tok
            if len(combined) > max_total_length:
                combined = combined[:max_total_length]
            else:
                pad_len = max_total_length - len(combined)
                combined = combined + [self.tokenizer.pad_token_id] * pad_len
            combined_tokens.append(combined)
        input_ids = torch.tensor(combined_tokens, dtype=torch.long, device=padded_src_audios.device)
        
        # Construct labels: set the text portion (the first t_len tokens) to ignore_index, keeping the speech tokens unchanged
        labels = input_ids.clone()
        for i in range(batch_size):
            if (self.use_text and not self.text_guide) or self.task =='asr':
                first_gen_pos = (input_ids[i] == self.text_gen_start_id).nonzero(as_tuple=True)[0].item()
            else:
                first_gen_pos = (input_ids[i] == self.speech_gen_start_id).nonzero(as_tuple=True)[0].item()
            labels[i, :first_gen_pos] = self.ignore_index
        labels[input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def freeze_encoder(self):
        freeze_model(self.encoder)        

    # Override state_dict method to return only the llm part's parameters
    def state_dict(self, *args, **kwargs):
        return self.llm.state_dict(*args, **kwargs)

    # Override save_pretrained method to save only the llm part
    def save_pretrained(self, save_directory, **kwargs):
        self.llm.save_pretrained(save_directory, **kwargs)

############################################
# Arguments and Main Function
############################################

@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the data."})
    use_text: bool = field(default=False, metadata={"help": "Whether to use instruction."})
    task: str = field(default=None, metadata={"help": "Task type."})
    text_guide: bool = field(default=False, metadata={"help": "Whether to use text guide."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(default=None, metadata={"help": "Integration to report results."})
    run_name: Optional[str] = field(default=None, metadata={"help": "The name of the run for logging."})
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})
    use_lora: bool = field(default=False)

def main():
    default_config_file = 'config_ata.json'

    import argparse
    parser_json = argparse.ArgumentParser()
    parser_json.add_argument('--config', type=str, default=default_config_file, help="Path to the config JSON file")
    json_args = parser_json.parse_args()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        default_config_file = json_args.config
        print(f"[INFO] Loaded config from {json_args.config}")
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
    
    is_main_process = training_args.local_rank in [-1, 0]
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(project="llasa", config=training_args.to_sanitized_dict(), name=training_args.run_name)
    print(model_args.llm_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype='auto',
        cache_dir=model_args.cache_dir,
    )
    config = transformers.AutoConfig.from_pretrained(model_args.llm_model_name_or_path)
    device_id = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Import the speech codec model, for example XCodec2Model
    # from xcodec2.modeling_xcodec2 import XCodec2Model
    # model_path = "HKUSTAudio/xcodec2"
    # Codec_model = XCodec2Model.from_pretrained(model_path)
    sys.path.append('/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/xcodec2')
    from vq_process import extract_vq_code_for_offline_training as Codec_model
    
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=64,                      
            lora_alpha=128,           
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters:", trainable_params)

    data_split = load_dataset(
        'parquet',
        data_files={
            'train': [
                '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset/*.parquet',
            ]
        },
        split='train',
    )
    valid = True
    if valid:
       test_data_split = load_dataset(
            'parquet',
            data_files={
                'train': [
                    '/mnt/bn/tanman-yg/chenqi/datas/InstructSpeech_Dataset_eval/chunk_valid.parquet',
                ]
            },
            split='train',
        )

    train_dataset_raw = data_split
    from torch.utils.data import Subset
    # Instantiate custom dataset (pass in tokenizer for prompt construction and tokenization)
    train_dataset = WaveDataset(train_dataset_raw, sampling_rate=16000, tokenizer=tokenizer, use_text=data_args.use_text, task=data_args.task, text_guide=data_args.text_guide)
    test_dataset = WaveDataset(test_data_split, sampling_rate=16000, tokenizer=tokenizer, use_text=data_args.use_text, task=data_args.task, text_guide=data_args.text_guide)
    test_dataset = Subset(test_dataset, list(range(500)))

    lwc_model = llm_with_codec_model(config, model, Codec_model, tokenizer, use_text=data_args.use_text, task=data_args.task, text_guide=data_args.text_guide)
    lwc_model = lwc_model.to(device)
    # lwc_model.freeze_encoder()
    trainer = Trainer(
        model=lwc_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=pad_audio_batch
    )
    # __import__('ipdb').set_trace()
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
