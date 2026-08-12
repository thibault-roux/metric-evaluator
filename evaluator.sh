#!/bin/bash
#SBATCH --job-name=evaluator
#SBATCH --output=out/evaluator_%j.out
#SBATCH --error=out/evaluator_%j.err
#SBATCH --account=uniphore
#SBATCH --partition=gpu
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=4
#noSBATCH --mem=32G
#SBATCH --time=01:00:00

conda activate hats   # or your env name
python evaluator.py

# multilingual-e5-large-instruct 3593444
# Qwen3-Embedding-8B 3593480