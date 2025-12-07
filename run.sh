#!/bin/bash
#SBATCH --gres=gpu:1                   # Request 4 GPUs
#SBATCH --nodelist=gnode017            # Request specific node (optional)
#SBATCH --cpus-per-gpu=16              # Number of CPUs per GPU
#SBATCH --time=4-00:00:00              # Max time (4 days)
#SBATCH --output=co3d_bench_progress.txt    # Output log file
#SBATCH --mail-user=aditya.vadali@research.iiit.ac.in  # Email for notifications
#SBATCH --mail-type=ALL                # Notify on BEGIN, END, FAIL, etc.

# Activate your Python environment
source ~/.bashrc
conda activate splatt3r

# Navigate to your project directory
cd ~/splatt3r-MR-Project

python model_replacement_test.py
