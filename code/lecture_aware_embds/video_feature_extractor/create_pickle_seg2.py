import pickle as pkl
import os.path
import os
import argparse
import json

from tqdm import tqdm

from os.path import join, isfile
from glob import glob

# from sentence_transformers import SentenceTransformer, util

import numpy as np
import subprocess

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

ocr_dir = '/ssd_scratch/cvit/darshan/OCR'

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
    return float(result.stdout)

def toFFMPEGtime(t):
    ss, ms = divmod(t*1000, 1000)
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)

    return "{:02d}:{:02d}:{:02d},{:03d}".format(int(hh), int(mm), int(ss), int(ms))

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

def getOCR(course_name, vid_name, st, et):
    lec_name = "-".join(vid_name.split('-')[:-1])
    split_num = int(vid_name.split('-')[-1].replace('.mp4', ''))

    # json_data = glob(join(ocr_dir, course_name, lec_name, '*.json'))[0]
    # json_data = open(json_data, 'r')
    # json_data = json.load(json_data)
    # fps = round(json_data['frame_metadata']['True FPS'], 2)

    ocr_text = ""
    ocr_frame_num = None
    ocr_frame_ts = None
    ocr_lec_name = lec_name

    for fl in glob(join(ocr_dir, course_name, lec_name, '*.json')):
        json_data_fl_ref = open(fl, 'r')
        json_data_fl = json.load(json_data_fl_ref)
        json_data_fl_ref.close()
        fps = json_data_fl['frame_metadata']['True FPS']
        frame_num = json_data_fl['frame_metadata']['Frame number']

        if frame_num == 1:
            continue

        frame_ts = round(frame_num / fps, 3)
        
        if st <= frame_ts and frame_ts <= et:
            if 'fullTextAnnotation' in json_data_fl:
                ocr_text = json_data_fl['fullTextAnnotation']['text']
                ocr_frame_num = frame_num
                ocr_frame_ts = frame_ts
                break
    
    return ocr_text, ocr_frame_num, ocr_frame_ts, ocr_lec_name

data  = []

folder_list = []

for fl in glob(join(base_dir, '*')):
	if 'mit' in fl:
		folder_list.append(fl)

folder_list.sort()
print(folder_list)

# model_qa = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
# model_ss = SentenceTransformer('all-mpnet-base-v2')

count_match = 0
total_count = 0

for folder in folder_list:
	count = 0

	# print("Inside - ", folder)
	course_name = folder.split('/')[-1]
	print("Inside course -", course_name)

	with open(join(folder, 'combined.txt'), 'r') as text_file:

		lines = text_file.readlines()

		line_lst = []

		# for line in lines:
		# 	l = line.strip()
		# 	line_lst.append(l)

		for line in tqdm(lines):
			l = line.strip()
			count += 1
			features = {}
			vid_name = l.split(delimiter)[0]
			subtitle = l.split(delimiter)[1]
			subtitle = " ".join(subtitle.split())
			st = getTime(l.split(delimiter)[2])
			et = getTime(l.split(delimiter)[3])
			features_name = vid_name.replace('.mp4', '.npy')

			# if isfile(join(base_dir, 'features', '2d', features_name)):
			if isfile(join(base_dir, 'features', '2d', features_name)) and isfile(join(base_dir, 'features', '3d', features_name)):

				two_d = np.load(join(base_dir, 'features', '2d', features_name)) 
				three_d = np.load(join(base_dir, 'features', '3d', features_name))

				#if two_d.shape == (0, 2048):	
				if two_d.shape == (0, 2048) or three_d.shape == (0, 2048):
					print("True")
					continue
				
				# if mean-pooling
				# two_d = two_d.mean(axis = 0)
				# three_d = three_d.mean(axis = 0)
				
				# if max-pooling
				two_d = two_d.max(axis = 0)
				three_d = three_d.max(axis = 0)
			
				features['2d'] = two_d
				features['3d'] = three_d
				features['caption'] = subtitle
				features['id'] = vid_name.replace('.mp4', '')
				# features['vid_duration'] = get_length(join(folder, 'splits_vid', vid_name))
				features['vid_duration'] = et - st
				features['st'] = st
				features['et'] = et
				features['course_name'] = course_name

				# features['emb_qa'] = model_qa.encode(subtitle)
				# features['emb_ss'] = model_ss.encode(subtitle)

                # retrieving OCR data
				lec_name = "-".join(vid_name.split('-')[:-1])
				
				ocr_data = getOCR(course_name, vid_name, float(st), float(et))

				features['ocr_text'] = " ".join(ocr_data[0].split())
				features['ocr_frame_num']  = ocr_data[1]
				features['ocr_frame_ts'] = ocr_data[2]
				features['ocr_lec_name'] = ocr_data[3]

				data.append(features)
				count_match += 1

	print("Count = ", count)
	total_count += count

print("Count match = ", count_match)
print("Total count = ", total_count)

with open(join(base_dir, 'v1_1.pkl'), 'wb') as handle:
	pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)	
