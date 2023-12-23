#!/bin/bash
#SBATCH -A $USER

module load ffmpeg/4.4.1

echo "Trimming started"

python cut_intro.py

echo "DONE successfully"
