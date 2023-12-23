#!/bin/bash
#SBATCH -A $USER

module load ffmpeg/4.4.1

echo "Concat started"

python concat_videos.py

echo "DONE successfully"
