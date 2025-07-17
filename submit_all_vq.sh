#!/bin/bash

STEP=30000
SPLIT="train"
TARGET=$1  # 从命令行传入：csv 或 real
SCRIPT="run_vq_extract.sub"

if [ "$TARGET" != "csv" ] && [ "$TARGET" != "real" ]; then
  echo "[ERROR] Usage: bash submit_all_vq.sh [csv|real]"
  exit 1
fi

if [ "$TARGET" = "csv" ]; then
  CSV_PATH="/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/${SPLIT}_vc_spk_filtered.csv"
  total=$(($(wc -l < "$CSV_PATH") - 1))  # 减去header
elif [ "$TARGET" = "real" ]; then
  AUDIO_DIR="/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset"
  total=$(find "$AUDIO_DIR" -type f -name "*.wav" | wc -l)
fi

echo "[INFO] Total $TARGET items: $total"

# 提交任务
for ((start=0; start<$total; start+=STEP)); do
    end=$((start + STEP))
    if [ $end -gt $total ]; then
        end=$total
    fi
    echo "[INFO] Submitting job: $start -> $end"
    sbatch "$SCRIPT" --start_index "$start" --end_index "$end" --split "$SPLIT" --target "$TARGET"
done
