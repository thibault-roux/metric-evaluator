# #!/bin/bash
# #SBATCH --account satcon
# #SBATCH --job-name 🍀kfold_simeval
# #SBATCH --partition=gpu
# #SBATCH --gpus=v100:1
# #SBATCH --ntasks=1
# #SBATCH --nodes=1
# #SBATCH --mem=64G
# #SBATCH --cpus-per-task=20
# #SBATCH --time=5:00:00
# #SBATCH --output=logs/kfold_simeval.%j.out
# #SBATCH --error=logs/kfold_simeval.%j.err



  python kfold_simeval.py \
    --dataset_path datasets/hats.txt \
    --output_dir results/kfold \
    --dtype bfloat16 \
    --batch_size 8
