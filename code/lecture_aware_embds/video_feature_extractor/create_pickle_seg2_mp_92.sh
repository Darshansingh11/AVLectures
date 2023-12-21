#!/bin/bash
#SBATCH -A darshan.singh
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=INFINITE
#SBATCH --mail-user=darshans012@gmail.com
#SBATCH --mail-type=ALL

module load ffmpeg/4.4.1

echo "Pickling started"

python create_pickle_seg2_mp.py --base_dir='/ssd_scratch/cvit/darshan/dataset_MITOCW_v1' --file_name='m080_mit103'

echo "Pickling ended successfully"
