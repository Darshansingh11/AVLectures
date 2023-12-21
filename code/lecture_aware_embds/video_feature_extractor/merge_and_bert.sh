#!/bin/bash
#SBATCH -A darshan.singh
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=INFINITE
#SBATCH --mail-user=darshans012@gmail.com
#SBATCH --mail-type=ALL

echo "started"

python merge_and_bert.py

echo "Finished successfully"
