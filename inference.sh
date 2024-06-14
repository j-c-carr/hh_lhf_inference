#!/bin/bash
#SBATCH --output=logs/job-output-%j_pythia28_rtp_with_tox.txt
#SBATCH --error=logs/job-error-%j_pythia28_rtp_with_tox.txt
#SBATCH --mem=64Gb
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mail-user=jonathan.colaco-carr@mila.quebec

module load python/3.8
module load cuda/11.7

# Activate the virtual environment (same as used for direct preference optimization)
dpo_dir=$HOME/courses/c597/direct-preference-optimization
source $dpo_dir/venv/bin/activate

# Run evaluation script
python main.py

