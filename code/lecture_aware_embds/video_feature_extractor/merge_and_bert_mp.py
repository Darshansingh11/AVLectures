import pickle as pkl
import os.path
import os
import argparse
import json

from tqdm import tqdm

from os.path import join, isfile
from glob import glob

from sentence_transformers import SentenceTransformer, util

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import numpy as np
import subprocess
import copy

base_dir = '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1_4s8s'

print("Loading BERT models")

# model_qa = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
model_ss = SentenceTransformer('all-mpnet-base-v2')

print("Done model loading")

p_list = []

for p in glob(join(base_dir, '*')):
    p_list.append(p)

p_list.sort()

# p_list = [join(base_dir, 'v1_2d3dOCR_4_8.pkl')]

print(p_list)

data = []

def do_job(f):
    features = f.copy()

    caption = features['caption']

    # features['emb_qa'] = model_qa.encode(caption)
    features['emb_ss'] = model_ss.encode(caption)

    ocr_text = f['ocr_text']
    # features['ocr_emb_qa'] = model_qa.encode(ocr_text)
    features['ocr_emb_ss'] = model_ss.encode(ocr_text)

    return features

for pkl_fl in p_list:
    b = pkl.load(open(pkl_fl, 'rb'))

    p = ProcessPoolExecutor(30)

    futures = [p.submit(do_job, li) for li in b]
    x = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    data.extend(x)

    # for f in tqdm(b):
    #     features = f.copy()

    #     caption = features['caption']

    #     # features['emb_qa'] = model_qa.encode(caption)
    #     features['emb_ss'] = model_ss.encode(caption)

    #     ocr_text = f['ocr_text']
    #     # features['ocr_emb_qa'] = model_qa.encode(ocr_text)
    #     features['ocr_emb_ss'] = model_ss.encode(ocr_text)

    #     data.append(features)


print(len(data))

with open(join(base_dir, 'datasetv1_4s8s_2d3dOCRBERT.pkl'), 'wb') as handle:
	pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)