#!/bin/bash
#SBATCH -A darshan.singh
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=INFINITE
#SBATCH --mail-user=darshans012@gmail.com
#SBATCH --mail-type=ALL


echo "Pickling started"

python create_pickle_seg2_mp.py --base_dir='/ssd_scratch/cvit/darshan/dataset_MITOCW_v1' --file_name='m050_m156'

echo "Pickling ended successfully"
