#!/bin/bash
#SBATCH --job-name=evaluator
#SBATCH --output=out/evaluator_%j.out
#SBATCH --error=out/evaluator_%j.err
#SBATCH --account=satcon
#SBATCH --partition=gpu
# #SBATCH --gpus=rtx3090:1
# #SBATCH --qos=prio-opportuneness_autodialling-14d
#SBATCH --gpus=h100:1 # I USE H100 FOR SOME QWEN-8B MODELS
#SBATCH --cpus-per-task=4
#noSBATCH --mem=32G
#SBATCH --time=12:00:00

conda activate bertscore   # or your env name
python evaluator_bertscore.py
