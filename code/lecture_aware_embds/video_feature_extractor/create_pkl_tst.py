import pickle as pkl
import os.path
import os
import argparse

from os.path import join, isfile
from glob import glob

import numpy as np

# The default path of DataSubset is '/ssd_scratch/cvit/AVLectures/DataSubset' (also referred to as base_dir)
# base_dir should also contain a directory inside it with the name "features" which inturn contains two directories called "2d" and "3d" which contains 2d and 3d video features respectively. 
# If you want to change the default path, then you can do it using the optional '--base_dir' argument
# After executing this code, a pickle file called 'avl.pkl' would be created inside the base_dir.

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", type=str, required=False, help="Path of DataSubset", default = '/ssd_scratch/cvit/AVLectures/DataSubset')
args = parser.parse_args()

base_dir = args.base_dir
print("Base Directory:")
print(base_dir)

data  = []

folder_list = []

for fl in glob(join(base_dir, '*')):
	# if ('mit011' in fl) or ('mit012' in fl):
	# 	continue
	# if 'mit' in fl:
	# 	folder_list.append(fl)

	if ('mit006' in fl):
		folder_list.append(fl)

folder_list.sort()
print(folder_list)

count_match = 0
total_count = 0

features = {}

for folder in folder_list:
	count = 0

	print("Inside - ", folder)

	with open(join(folder, 'combined.txt'), 'r') as text_file:

		lines = text_file.readlines()

		for line in lines:
			count += 1
			
			vid_name = line.split('|')[0]
			subtitle = line.split('|')[1]
			subtitle = " ".join(subtitle.split())

			features_name = vid_name.replace('.mp4', '.npy')
			
			if isfile(join(base_dir, 'features', '2d', features_name)) and  isfile(join(base_dir, 'features', '3d', features_name)):

				two_d = np.load(join(base_dir, 'features', '2d', features_name))
				three_d = np.load(join(base_dir, 'features', '3d', features_name))

				if two_d.shape == (0, 2048) or three_d.shape == (0, 2048) or len(subtitle) == 0:
					print("True")
					continue

				# two_d = two_d.mean(axis = 0)
				# three_d = three_d.mean(axis = 0)
				
				two_d = two_d.max(axis = 0)
				three_d = three_d.max(axis = 0)
			
				# features['2d'] = two_d
				# features['3d'] = three_d
				# features['caption'] = subtitle
				features[features_name.replace('.npy', '')] = subtitle

				# data.append(features)
				count_match += 1

	print("Count = ", count)
	total_count += count

print("Count match = ", count_match)
print("Total count = ", total_count)

with open(join(base_dir,  'm6_pysd_subs.pkl'), 'wb') as handle:
	pkl.dump(features, handle, protocol=pkl.HIGHEST_PROTOCOL)	
