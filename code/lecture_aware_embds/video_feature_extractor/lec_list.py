import pickle as pkl
import os.path
import os
import argparse
import json

from tqdm import tqdm

from os.path import join, isfile
from glob import glob

import numpy as np
import subprocess

base_dir = '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_20s25s'

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


base_pkl = pkl.load(open(join(base_dir, 'dataset_v1_20s25s_2d3dOCRBERT.pkl'), 'rb'))


d = []

for f in tqdm(base_pkl):
    vid_name = "-".join(f['id'].split('-')[:-1])
    if vid_name not in d:
        d.append(vid_name)

with open(join(base_dir, 'dataset_v1_leclist_20s25s.pkl'), 'wb') as handle:
	pkl.dump(d, handle, protocol=pkl.HIGHEST_PROTOCOL)	

print(len(d))
