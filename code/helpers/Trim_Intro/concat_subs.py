import subprocess, traceback
import os
import argparse

import cv2

import pickle as pkl

from os.path import join
from glob import glob
from pathlib import Path
from tqdm import tqdm

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_length_cv2(vid_path):
    video = cv2.VideoCapture(vid_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return round(frame_count / fps, 2)

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

def toFFMPEGtime(t):
    ss, ms = divmod(t*1000, 1000)
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)

    return "{:02d}:{:02d}:{:02d},{:03d}".format(int(hh), int(mm), int(ss), int(ms))


def concat_subtitles(course, out_sub_filename, sub_files):
    num_sub_files = len(sub_files)
    offset = 0
    cnt = 1

    for sub in sub_files:
        with open(join(course, 'subtitles', sub), 'r') as orig_sub, open(join(course, 'concatenated_subtitles', out_sub_filename), 'a') as concat_sub:
            lines = orig_sub.readlines()
            endHere = False
            last_arrow_idx = None
            for i, line in reversed(list(enumerate(lines))):
                l = line.strip()
                if '-->' in l:
                    last_arrow_idx = i
                    break

            for i, line in enumerate(lines):
                l = line.strip()

                if i < len(lines) - 1:
                    next_line =  lines[i + 1].strip()

                    if '-->' in next_line:
                        concat_sub.write('{}\n'.format(str(cnt)))
                        cnt += 1
                        endHere = False
                        continue

                if '-->' in l:
                    ts = l.split(' ')
                    st, en = ts[0].strip(), ts[2].strip()
                    st_shifted = toFFMPEGtime(getTime(st) + offset)
                    en_shifted = toFFMPEGtime(getTime(en) + offset)

                    if i == last_arrow_idx:
                        last_arrow_time = toFFMPEGtime(get_length_cv2(join(course, 'videos', sub.replace('.srt', '.mp4'))) + offset)
                        concat_sub.write('{} --> {}\n'.format(st_shifted, last_arrow_time))
                    else:
                        concat_sub.write('{} --> {}\n'.format(st_shifted, en_shifted))
                    endHere = True
                else:
                    concat_sub.write('{}'.format(line))
                    endHere = False
            
            if endHere:
                concat_sub.write('\n')
            
            offset += get_length_cv2(join(course, 'videos', sub.replace('.srt', '.mp4')))

sub_files = ['MIT8_01F16_L19v01_360p.srt', 'MIT8_01F16_L19v02_360p.srt', 'MIT8_01F16_L19v03_360p.srt', 
            'MIT8_01F16_L19v04_360p.srt', 'MIT8_01F16_L19v05_360p.srt', 'MIT8_01F16_L19v06_360p.srt', 
            'MIT8_01F16_L19v07_360p.srt']

concat_subtitles('/ssd_scratch/cvit/darshan/dataset_MITOCW_v1/mit032', 'L19.srt', sub_files)