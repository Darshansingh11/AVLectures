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

delimiter = "@#@"

data  = []

folder_list = []

def getTime(t):
	h, m, sms = t.split(":")
	if ',' in sms: # Example t = '00:00:03,980'
		s, ms = sms.split(",")
	elif '.' in sms: # Example t = '00:00:03.980'
		s, ms = sms.split(".")
	else: # Example t = '00:00:03'
		s = sms
		ms = '0'
	tm = 3600 * int(h) + 60 * int(m) + int(s) + int(ms.ljust(3, '0'))/1000
	return tm

for fl in glob(join(base_dir, '*')):
	# if ('mit011' in fl) or ('mit012' in fl):
	# 	continue
	if 'mit' in fl:
		folder_list.append(fl)

folder_list.sort()
print(folder_list)

count_match = 0
total_count = 0

for folder in folder_list:
	count = 0

	course_name = folder.split('/')[-1]
	print("Inside course -", course_name)

	# for glob(join(folder, 'subtitles'))

	with open(join(folder, 'combined.txt'), 'r') as text_file:

		lines = text_file.readlines()
		lines = sorted(lines, key = lambda line: (line.split(delimiter))[0])

		for i in range(len(lines)):
			l = lines[i].strip()
			count += 1
			features = {}

			present_vid = lines[i].split(delimiter)[0]  # example : "MIT6_042JF10_lec17_300k-00000.mp4"
			present_vid_name = '-'.join((present_vid.split('-'))[:-1]) # example : "MIT6_042JF10_lec17_300k"
			present_vid_id = (present_vid.split('-'))[-1] # example : "00000.mp4"
			present_vid_id = int(present_vid_id.replace('.mp4', '')) # example : 0
			present_vid_subtitle = lines[i].split(delimiter)[1]
			present_vid_subtitle = " ".join(present_vid_subtitle.split())

			prev_vid = ""
			next_vid = ""

			subtitle = ""

			if i > 0:
				prev_vid = lines[i - 1].split(delimiter)[0]
			if i < len(lines) - 1:
				next_vid = lines[i + 1].split(delimiter)[0]


			if prev_vid != '':

				prev_vid_name = '-'.join((prev_vid.split('-'))[:-1])
				prev_vid_id = (prev_vid.split('-'))[-1]
				prev_vid_id = int(prev_vid_id.replace('.mp4', ''))
				prev_vid_subtitle = lines[i - 1].split(delimiter)[1]
				prev_vid_subtitle = " ".join(prev_vid_subtitle.split())

				if present_vid_name == prev_vid_name and prev_vid_id ==  present_vid_id - 1:
					subtitle = prev_vid_subtitle + " "
			
			subtitle = subtitle + present_vid_subtitle

			if next_vid != '':
				
				next_vid_name = '-'.join((next_vid.split('-'))[:-1])
				next_vid_id = (next_vid.split('-'))[-1]
				next_vid_id = int(next_vid_id.replace('.mp4', ''))
				next_vid_subtitle = lines[i + 1].split(delimiter)[1]
				next_vid_subtitle = " ".join(next_vid_subtitle.split())

				if present_vid_name == next_vid_name and next_vid_id ==  present_vid_id + 1:
					subtitle = subtitle + " " + next_vid_subtitle		

			features_name = present_vid.replace('.mp4', '.npy')
			st = getTime(l.split(delimiter)[2])
			et = getTime(l.split(delimiter)[3])
			
			if isfile(join(base_dir, 'features', '2d', features_name)) and  isfile(join(base_dir, 'features', '3d', features_name)):

				two_d = np.load(join(base_dir, 'features', '2d', features_name))
				three_d = np.load(join(base_dir, 'features', '3d', features_name))

				if two_d.shape == (0, 2048) or three_d.shape == (0, 2048):
					print("True")
					continue

				# two_d = two_d.mean(axis = 0)
				# three_d = three_d.mean(axis = 0)
				
				two_d = two_d.max(axis = 0)
				three_d = three_d.max(axis = 0)
			
				features['2d'] = two_d
				features['3d'] = three_d
				features['caption'] = subtitle
				features['id'] = present_vid.replace('.mp4', '')
				features['st'] = st
				features['et'] = et

				features['vid_duration'] = et - st

				features['course_name'] = course_name

				data.append(features)
				count_match += 1

	print("Count = ", count)
	total_count += count

print("Count match = ", count_match)
print("Total count = ", total_count)

with open(join(base_dir,  'seg_10s15s_2d3dprevnext.pkl'), 'wb') as handle:
	pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)	