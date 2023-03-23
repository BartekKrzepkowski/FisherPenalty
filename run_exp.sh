#!/bin/bash
#SBATCH --job-name=test
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --partition=batch

srun run_exp.py