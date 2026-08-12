#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --output=out/llm_eval_%j.out
#SBATCH --error=out/llm_eval_%j.err
#SBATCH --account=uniphore
#SBATCH --partition=gpu
# #SBATCH --gpus=rtx3090:1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#noSBATCH --mem=32G
#SBATCH --time=03:00:00

conda activate hats   # or your env name, previously "hats"
python llm_eval.py

# Qwen 3593202
# google/gemma-4-31B 3593482