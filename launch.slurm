#!/bin/bash
#SBATCH --job-name=evaluator
#SBATCH --time=36:30:00
#SBATCH --output=output.txt
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END,FAIL
#notSBATCH --exclude=idyie,alpos,aura 




#python evaluator.py
#python agreement.py
python complete_evaluator.py
