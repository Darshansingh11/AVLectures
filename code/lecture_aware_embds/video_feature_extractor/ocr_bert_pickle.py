import pickle as pkl
import os.path
import os
import argparse
import json

from tqdm import tqdm

from os.path import join, isfile
from glob import glob

from sentence_transformers import SentenceTransformer, util

import numpy as np
import subprocess


base_dir = '/ssd_scratch/cvit/darshan/segmentation_dataset_v1_10s15s'


delimiter = "@#@"

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


model_qa = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
model_ss = SentenceTransformer('all-mpnet-base-v2')

base_pkl = pkl.load(open(join(base_dir, 'old_seg_10s15s_2d3dBERTOCR.pkl'), 'rb'))

data = []

for f in tqdm(base_pkl):
    features = f.copy()
    ocr_text = f['ocr_text']
    features['ocr_emb_qa'] = model_qa.encode(ocr_text)
    features['ocr_emb_ss'] = model_ss.encode(ocr_text)


    data.append(features)


with open(join(base_dir, 'seg_10s15s_2d3dOCRBERT.pkl'), 'wb') as handle:
	pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)	