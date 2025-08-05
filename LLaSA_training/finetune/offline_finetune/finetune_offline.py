import os
import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
from dataclasses import dataclass, field
from typing import Optional, Literal
import sys
import transformers
import wandb
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import random
from datasets import load_dataset
from functools import partial
from peft import LoraConfig, get_peft_model
 
 
@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

 
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})
    text_guidance: bool = field(default=False, metadata={"help": "Whether to use text guidance."})
    
 
@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler to use."})
    # evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="steps", metadata={"help": "The evaluation strategy to use."})
    eval_steps: int = field(default=1, metadata={"help": "Evaluate every X updates steps."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use lora."})
    

class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, text_guidance=False):
 
        memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')
        instruct_path = os.path.join(data_path, f'{split}_data.csv')

        self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.length = self.input_ids.shape[0]
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer
        self.instuct = pd.read_csv(instruct_path, header=0)['trg_instruct'].tolist()
        self.text_guidance = text_guidance
   
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
 
        self.max_length = 2048
        self.ignore_index = -100  
        
    def __len__(self):
        return self.length

    def replace_tagged_token(self, token_list, target_token, new_sequence):
        idx = token_list.index(target_token)
        return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        instruct = self.instuct[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        speech_gen_positions = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0]
        text_gen_positions = (input_ids == self.text_generation_start_id).nonzero(as_tuple=True)[0]
        speech_gen_idx = speech_gen_positions[0].item()

        try:
            speech_gen_end_idx = (input_ids == self.speech_generation_end_id).nonzero(as_tuple=True)[0].item()
        except Exception as e:
            print(f"maybe Error in speech_gen_end_idx: {e}")
            # speech_gen_end_idx = len(input_ids) - 1
            speech_gen_end_idx = 2048
        # __import__('remote_pdb').set_trace()

        src_speech = input_ids[:text_gen_positions]
        text = input_ids[text_gen_positions:speech_gen_idx]
        gen_speech = input_ids[speech_gen_idx: speech_gen_end_idx + 1]
        
        text_front = False
        if self.text_guidance:
            text_gen_idx = text_gen_positions[0].item()
            try:
                src_speech = input_ids[:text_gen_idx]
                gen_speech = input_ids[speech_gen_idx: speech_gen_end_idx + 1]
                text_gen_end_idx = (input_ids == self.text_generation_end_id).nonzero(as_tuple=True)[0].item()
                gen_text = input_ids[text_gen_idx:text_gen_end_idx + 1]
            except Exception as e:
                print(f"Error in text_gen_end_idx: {e}")
                text_gen_end_idx = 2048

            # __import__('ipdb').set_trace()
            system_content = (
                "You are an expert speech assistant. Your task is to generate an accurate and clear transcription of the input speech, and follow the given instruction to produce the appropriate speech."
                if text_front else
                "You are an expert speech assistant. Your task is to follow the given instruction to produce the appropriate speech, and generate an accurate and clear transcription of the input speech."
            )
            chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "The transcription of input speech is: <|TEXT_GENERATION_START|>, the converted speech is <|SPEECH_GENERATION_START|>" if text_front else "The converted speech is: <|SPEECH_GENERATION_START|>, The transcription of input speech is: <|TEXT_GENERATION_START|>"}
            ]
        else:
            chat = [
                {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)
    
        ids = self.replace_tagged_token(ids, self.speech_understanding_start_id, src_speech)
        ids = self.replace_tagged_token(ids, self.speech_generation_start_id, gen_speech)
        if self.text_guidance:
            ids = self.replace_tagged_token(ids, self.text_generation_start_id, gen_text)

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)
        try:
            speech_gen_idx_in_input = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0].item()
            labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]

            if self.text_guidance and text_front:
                text_gen_idx_in_input = (input_ids == self.text_generation_start_id).nonzero(as_tuple=True)[0].item()
                labels[text_gen_idx_in_input:] = input_ids[text_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
            labels  = input_ids 

        attention_mask = (input_ids != self.pad_token_id).long()
 
        labels[input_ids == self.pad_token_id] = self.ignore_index
        
        input_ids = self.pad_sequence(input_ids, self.max_length, value=self.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, self.max_length, value=0)
        labels = self.pad_sequence(labels, self.max_length, value=self.ignore_index)

        return {
            'input_ids': list(input_ids),
            'labels': list(labels),
            'attention_mask': list(attention_mask)
        }
 
def main():
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # Load arguments from the specified JSON file
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Attempt to load arguments from the default 'config.json' file
        default_config_file = 'config.json'
 
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
     
    is_main_process = training_args.local_rank in [-1, 0]
 
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",  
            config=training_args.to_sanitized_dict(),
            name=Path(training_args.run_name).stem
        )
 
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype='auto',
        cache_dir=model_args.cache_dir,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=8,                      
            lora_alpha=32,           
            target_modules=["q_proj", "v_proj"],   
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA微调模型参数信息：")
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters:", trainable_params)

    train_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='train',
        tokenizer=tokenizer,
        text_guidance=data_args.text_guidance
    )
    eval_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='valid',
        tokenizer=tokenizer,
        text_guidance=data_args.text_guidance
    ) if os.path.exists(os.path.join(data_args.data_path, 'valid_input_ids.memmap')) else None
    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if is_main_process:
        trainer.add_callback(transformers.integrations.WandbCallback())

    trainer.train()
 
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":

    # path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin/train_instruct.csv'
    main()
