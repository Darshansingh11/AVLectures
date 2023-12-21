import pickle as pkl
import os.path
import os
import argparse
import json

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

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
parser.add_argument("-f", "--file_name", type=str, required=False, help="", default = 'm')
args = parser.parse_args()

base_dir = args.base_dir
print("Base Directory:")
print(base_dir)

f_name = args.file_name
f_name = "20_25"
print(f_name)

delimiter = "@#@"

ocr_dir = '/ssd_scratch/cvit/darshan/OCR/dataset_MITOCW_v1'

base_pkl = pkl.load(open('/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_20s25s/v1_1.pkl', 'rb'))

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


def do_job(l):

    course_name = l['course_name']

    if not os.path.isdir(join(ocr_dir, course_name)):
        return

    st = l['st']
    et = l['et']
    vid_name = l['id'] + '.mp4'

    lec_name = "-".join(vid_name.split('-')[:-1])

    ocr_data = getOCR(course_name, vid_name, float(st), float(et))

    l['ocr_text'] = " ".join(ocr_data[0].split())
    l['ocr_frame_num']  = ocr_data[1]
    l['ocr_frame_ts'] = ocr_data[2]
    l['ocr_lec_name'] = ocr_data[3]

    return l


count_match = 0
total_count = 0
# p = ThreadPoolExecutor(20)
p = ProcessPoolExecutor(19)

futures = [p.submit(do_job, li) for li in base_pkl]
x = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

# print(d_ocr[0])
# print(x)

with open(join(base_dir, 'v1_2d3dOCR_{}.pkl'.format(f_name)), 'wb') as handle:
	pkl.dump(x, handle, protocol=pkl.HIGHEST_PROTOCOL)	
