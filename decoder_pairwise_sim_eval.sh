#!/bin/bash
#SBATCH --account satcon
#SBATCH --job-name 🍀pairwise_sim_eval
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=logs/dec_pairwise_sim_eval.%j.out
#SBATCH --error=logs/dec_pairwise_sim_eval.%j.err



for num_layer in -1 -2 -3 -4 -5 -6 -7 -8 -9 -10
do
    echo "Evaluating with num_layer = $num_layer"

    python decoder_pairwise_sim_eval.py \
      --dataset_path datasets/hats-en.txt \
      --summary_csv shashi_analysis/summary_results_${num_layer}_en.csv \
      --dtype bfloat16 \
      --batch_size 8 \
      --certitudes 1.0 \
      --num_layer ${num_layer} \
      --min_votes 0

done