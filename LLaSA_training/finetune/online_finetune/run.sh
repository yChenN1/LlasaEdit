#!/bin/sh
sudo apt-get install jq -y
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16
export HF_DATASETS_CACHE=/mnt/bn/tanman-yg/chenqi/datas/.hf_dataset_cache
EXP_NAME=$1
MODEL_PATH="/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/Llasa-1B"

# 获取当前日期（MMDD）
TODAY=$(date +%m%d)

# 生成 output_dir
PREFIX="/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/LLaSA_training/exp/online/a2a/finetune"
OUTPUT_DIR="${PREFIX}/${TODAY}_a2a_${EXP_NAME}"

# 临时修改后的 config 文件路径
CONFIG_TEMPLATE=config_ata.json
CONFIG_PATH=config_ata_${EXP_NAME}.json

# 使用 jq 替换 llm_model_name_or_path 和 output_dir
jq \
  --arg model_path "$MODEL_PATH" \
  --arg output_dir "$OUTPUT_DIR" \
  '.llm_model_name_or_path = $model_path | .output_dir = $output_dir' \
  "$CONFIG_TEMPLATE" > "$CONFIG_PATH"


# Launch training
# torchrun --nproc_per_node=1 --master-port 10203 finetune_offline_w_rl.py
torchrun --nproc_per_node=8 --master-port 20227 finetune_online_ata.py --config "$CONFIG_PATH" | tee ${TODAY}_a2a_${EXP_NAME}.log


# export NCCL_CROSS_NIC=1
# export OMP_NUM_THREADS=1
# # export NCCL_ALGO=^Ring
# NUM_TOTAL_GPU=$((ARNOLD_WORKER_NUM*ARNOLD_WORKER_GPU))

# accelerate launch \
#     --num_machines $ARNOLD_WORKER_NUM \
#     --machine_rank $ARNOLD_ID \
#     --num_processes $NUM_TOTAL_GPU \
#     --main_process_ip $ARNOLD_WORKER_0_HOST \
#     --main_process_port $(echo $ARNOLD_WORKER_0_PORT | cut -d"," -f2) \
#     --dynamo_backend "no" \
#     --mixed_precision bf16 \
#     finetune_online_ata.py --config "$CONFIG_PATH" \
#     2>&1 | tee ${TODAY}_a2a_${EXP_NAME}.log