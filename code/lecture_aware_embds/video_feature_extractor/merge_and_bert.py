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
import copy

base_dir = '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_20s25s'

print("Loading BERT models")

# model_qa = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
model_ss = SentenceTransformer('all-mpnet-base-v2')

print("Done model loading")

p_list = ['/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_20s25s/v1_2d3dOCR_20_25_1.pkl', 
         '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_20s25s/v1_2d3dOCR_20_25_2.pkl']

# for p in glob(join(base_dir, '*')):
#     p_list.append(p)

p_list.sort()

# p_list = [join(base_dir, 'v1_2d3dOCR_4_8.pkl')]

print(p_list)

data = []

for pkl_fl in p_list:
    b = pkl.load(open(pkl_fl, 'rb'))

    for f in tqdm(b):
        if f is not None:
            features = f.copy()

            caption = features['caption']

            # features['emb_qa'] = model_qa.encode(caption)
            features['emb_ss'] = model_ss.encode(caption)

            ocr_text = f['ocr_text']
            # features['ocr_emb_qa'] = model_qa.encode(ocr_text)
            features['ocr_emb_ss'] = model_ss.encode(ocr_text)

            data.append(features)


print(len(data))

with open(join(base_dir, 'dataset_v1_20s25s_2d3dOCRBERT.pkl'), 'wb') as handle:
	pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
