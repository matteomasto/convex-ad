#!/bin/bash
#SBATCH --job-name=rec_job
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=log.out

module load cuda   # if required by cluster

python run_reconstruction.py