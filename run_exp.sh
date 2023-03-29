#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gpus=1
#SBATCH --qos=1gpu2d
#SBATCH --cpus-per-task=4  
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=common
#SBATCH --time=2-0

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $HOME/anaconda3/envs/fp

python3 -u run_exp.py