
import sys
sys.path.append('/mnt/fast/nobackup/users/yc01815/code/llasa/')
from trl1 import GRPOTrainer, GRPOConfig
import os
import json
import copy
import wandb
import torch
import random
import deepspeed
import transformers
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
from functools import partial
from datasets import load_dataset
from typing import Optional, Literal
from dataclasses import dataclass, field
from SpeechRLScorer import SpeechRLScorer
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_pt_utils import LabelSmoother

# @dataclass
# class CustomGRPOConfig:
#     reward_scaling: float = field(default=1.0)
#     mini_batch_size: int = field(default=1)
#     ppo_epochs: int = field(default=4)
#     total_steps: int = field(default=10000)
#     optimize_cuda_cache: bool = field(default=True)
#     init_kl_coef: float = field(default=0.2)
#     target_kl: float = field(default=0.1)


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

 
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})

 
@dataclass
class CustomGRPOArguments(GRPOConfig):
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
    evaluation_strategy: Literal["no", "steps", "epoch"] = field(default="no", metadata={"help": "The evaluation strategy to use."})
    reward_scaling: float = field(default=1.0)
    mini_batch_size: int = field(default=1)
    ppo_epochs: int = field(default=4)
    total_steps: int = field(default=10000)
    optimize_cuda_cache: bool = field(default=True)
    init_kl_coef: float = field(default=0.2)
    target_kl: float = field(default=0.1)
    num_generations: int = field(default=4)


class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
 
        memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')
        instruct_path = os.path.join(data_path, f'{split}_data.csv')

        self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.length = self.input_ids.shape[0]
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer
        self.instuct = pd.read_csv(instruct_path, header=0)['instruct'].tolist()
        self.transcription = pd.read_csv(instruct_path, header=0)['transcription'].tolist()

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

    def pad_sequence(self, sequence, max_length, value=0, padding_side='right'):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            if padding_side == 'right':
                return torch.cat([sequence, padding], dim=0)
            else:
                return torch.cat([padding, sequence], dim=0)
            
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        instruct = self.instuct[idx]
        transcription = self.transcription[idx]

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
 
        text_sequence = input_ids[:speech_gen_idx]
        speech_sequence = input_ids[speech_gen_idx : speech_gen_end_idx + 1]
 
        chat = [
            {"role": "user", "content": f"{instruct}:<|SPEECH_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
        
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True)
    
        ids = self.replace_tagged_token(ids, self.speech_understanding_start_id, text_sequence)
        # ids = self.replace_tagged_token(ids, self.speech_generation_start_id, speech_sequence)

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        try:
            speech_gen_idx_in_input = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0].item()
            labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
            labels  = input_ids 

        attention_mask = (input_ids != self.pad_token_id).long()
 
        labels[input_ids == self.pad_token_id] = self.ignore_index
        
        input_ids = self.pad_sequence(input_ids, self.max_length//2, value=self.pad_token_id, padding_side='left')
        attention_mask = self.pad_sequence(attention_mask, self.max_length//2, value=0, padding_side='left')
        labels = self.pad_sequence(labels, self.max_length//2, value=self.ignore_index, padding_side='left')

        return {
            'input_ids': list(input_ids),
            'labels': list(labels),
            'attention_mask': list(attention_mask),
            "expected_text": transcription,
            'prompt': list(input_ids),
            "promptabc": [
                {"role": "user", "content": "Convert the text to speech:" + "<|TEXT_UNDERSTANDING_START|><|TEXT_UNDERSTANDING_END|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ], 
        }


def reward_fn(samples, **kwargs):
    """
    samples: List of dicts, each with 'audio_path' 和 'expected_text'
    return: List of floats (reward per sample)
    """
    asr = kwargs.get("asr_model")  # 传入预加载模型以避免多次 load

    rewards = []
    for s in samples:
        audio_path = s["audio_path"]
        expected = s["expected_text"]
        gen_length = s["prompt_length"]   
        reward, pred_text, wer_val = asr.compute_reward(audio_path, expected, gen_length)
        rewards.append(reward)
    return rewards


def main():
    # 从JSON中分别加载配置
    with open("/mnt/fast/nobackup/users/yc01815/code/llasa/LLaSA_training/finetune/offline_finetune/config_grpo.json") as f:
        config = json.load(f)

    # 分别解析配置，避免冲突
    model_args, = HfArgumentParser(ModelArguments).parse_dict(config["model_args"])
    data_args, = HfArgumentParser(DataArguments).parse_dict(config["data_args"])

    # GRPO配置单独加载
    grpo_args = CustomGRPOArguments(**config["grpo_args"])

    # Wandb初始化
    is_main_process = grpo_args.local_rank in [-1, 0]
    if grpo_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",
            config=grpo_args.to_sanitized_dict(),
            name=grpo_args.run_name
        )


    training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
    # __import__('pdb').set_trace()

    # tokenizer/model/reward model加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=grpo_args.model_max_length,
        padding_side="left"
    )

    eos_token_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token_id = eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype='auto',
        cache_dir=model_args.cache_dir
    )
    model = model.float()
    
    # 如果使用 DeepSpeed，需要初始化 ref_model，防止空参数组报错
    # if grpo_args.deepspeed:
    #     ref_model = copy.deepcopy(model)
    #     ref_model.eval()
    #     for param in ref_model.parameters():
    #         param.requires_grad = False
    #         # 数据集初始化
    #     deepspeed_config = json.load(open(grpo_args.deepspeed))
    #     # 显式传空参数组，避免 torch.cat() 报错
    #     ref_model_engine, _, _, _ = deepspeed.initialize(
    #         model=ref_model,
    #         model_parameters=[],
    #         config=deepspeed_config
    #     )
        # ref_model = ref_model_engine  # 替换为包装后的模型

    train_dataset = TTSDataset(
            data_path=data_args.data_path,
            split='train',
            tokenizer=tokenizer
        )
    
    eval_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='val',
        tokenizer=tokenizer
    ) if os.path.exists(os.path.join(data_args.data_path, 'val_input_ids.memmap')) else None
    print(len(train_dataset), len(eval_dataset))

    data_collator = default_data_collator


    # __import__('ipdb').set_trace()
    SpeechRL = SpeechRLScorer()

    # 初始化GRPOTrainer
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        # ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=data_collator,
        reward_funcs=SpeechRL.compute_reward,
        # data_loader_kwargs={"worker_init_fn": seed_worker},
    )
   
    
    # __import__('ipdb').set_trace()

    if is_main_process:
        grpo_trainer.add_callback(transformers.integrations.WandbCallback())

 
    grpo_trainer.train()
 
    grpo_trainer.save_model(grpo_args.output_dir)
    tokenizer.save_pretrained(grpo_args.output_dir)


if __name__ == "__main__":
    main()
