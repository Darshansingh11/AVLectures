#!/bin/bash
#SBATCH -A darshan.singh
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04-00:00:00
#SBATCH --mail-user=darshans012@gmail.com
#SBATCH --mail-type=ALL

echo "Started"

python extract.py --csv=/ssd_scratch/cvit/AVL/data_subset_50s_60s/input_2d_m011_m012.csv --type=2d --batch_size=64 --num_decoding_thread=8

echo "Done successfully"
