import csv
import os
import argparse

from pathlib import Path
from os.path import join
from glob import glob

# The default path of DataSubset is '/ssd_scratch/cvit/AVLectures/DataSubset' (also referred to as base_dir)
# If you want to change the default path, then you can do it using the optional '--base_dir' argument
# After executing this code, two new CSV files would be created inside the base_dir
# 1. input_2d.csv   2. input_3d.csv
# Also empty directories called "features", "features/2d/", "features/3d/" will be created inside base_dir.

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", type=str, required=False, help="Path of DataSubset", default = '/ssd_scratch/cvit/AVLectures/DataSubset')
args = parser.parse_args()

base_dir = args.base_dir
print("Base Directory:")
print(base_dir)

# Create empty directories called "features/2d/" and "features/3d/"
Path(join(base_dir, "features", "2d")).mkdir(parents=True, exist_ok=True)
Path(join(base_dir, "features", "3d")).mkdir(parents=True, exist_ok=True)

fields = ['video_path', 'feature_path']

filename_2d = join(base_dir, 'input_2d.csv') # for extracting 2d features
filename_3d = join(base_dir, 'input_3d.csv') # for extracting 3d features

rows = []

folder_list = []

for fl in glob(join(base_dir, '*')):
	if 'mit' in fl:
		folder_list.append(fl)

folder_list.sort()
print(folder_list)

with open(filename_2d, 'w') as csvfile_2d, open(filename_3d, 'w') as csvfile_3d: 
	csvwriter_2d = csv.writer(csvfile_2d) 
	csvwriter_2d.writerow(fields) 

	csvwriter_3d = csv.writer(csvfile_3d) 
	csvwriter_3d.writerow(fields) 

	for folder in folder_list:
		rows_2d = []
		rows_3d = []
		count = 0
		print("Inside - ", folder)
		with open(join(folder, 'combined.txt'), 'r') as text_file:
			lines = text_file.readlines()
		
			for line in lines:
				count += 1
				vid_name = line.split('|')[0]
				vid_path = join(folder, vid_name)
				feature_name = vid_name.replace('.mp4', '.npy')
				feature_path_2d = join(base_dir, 'features', '2d', feature_name)
				feature_path_3d = join(base_dir, 'features', '3d', feature_name)
				rows_2d.append([vid_path, feature_path_2d])
				rows_3d.append([vid_path, feature_path_3d])

		print("Count =  ", count)
		csvwriter_2d.writerows(rows_2d)
		csvwriter_3d.writerows(rows_3d)