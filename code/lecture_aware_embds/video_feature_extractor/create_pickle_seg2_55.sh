#!/bin/bash
#SBATCH -A darshan.singh
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=INFINITE
#SBATCH --mail-user=darshans012@gmail.com
#SBATCH --mail-type=ALL

module load ffmpeg/4.4.1

echo "Pickling started"

python create_pickle_seg2.py --base_dir='/ssd_scratch/cvit/darshan/segmentation_dataset_v1_10s15s'

echo "Pickling ended successfully"
