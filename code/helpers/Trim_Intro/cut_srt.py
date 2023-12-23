import subprocess, traceback
import os
import argparse

from os.path import join
from glob import glob
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1'

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

def trim_subtitles(course, sub_file, start_ts, end_ts):

    with open(join(course, 'subtitles', sub_file), 'r') as orig_sub, open(join(course, 'trimmed_subtitles', sub_file), 'w') as trimmed_sub:
        start_copy = False
        end_copy = False
        count = 1

        lines = orig_sub.readlines()
        for i, line in enumerate(lines):
            l = line.strip()

            if '-->' in l and not start_copy:
                ts = l.split(' ')
                st, en = ts[0].strip(), ts[2].strip()
                if (getTime(st) - start_ts < 0) and (getTime(en) - start_ts <= 0):
                    start_copy = False
                else:
                    start_copy = True
                    trimmed_sub.write('{}\n'.format(str(count)))
                    count += 1

                    if getTime(st) != start_ts:
                        trimmed_sub.write('{} --> {}\n'.format('00:00:00,000', toFFMPEGtime(abs(getTime(en) - start_ts))))
                        continue
                        

            if start_copy and not end_copy:

                if i < len(lines) - 1:
                    next_line =  lines[i + 1].strip()
                    if '-->' in next_line:
                        next_line_st = next_line.split(' ')[0]
                        if getTime(next_line_st) >= end_ts:
                            end_copy = True
                            return None
                        trimmed_sub.write('{}\n'.format(str(count)))
                        count += 1
                        continue

                if '-->' in l:
                    ts = l.split(' ') 
                    st, en = ts[0].strip(), ts[2].strip()

                    st_shifted = toFFMPEGtime(getTime(st) - start_ts)
                    en_shifted = toFFMPEGtime(getTime(en) - start_ts)
                    trimmed_sub.write('{} --> {}\n'.format(st_shifted, en_shifted))
                else:
                    trimmed_sub.write(line)



course_list = []

for course in glob(join(base_dir, '*')):
     if 'mit' in course:
          course_list.append(course)

course_list.sort()

for course in course_list:
    print(course)

trim_subtitles('/ssd_scratch/cvit/darshan/dataset_MITOCW_v1/mit001', 'ocw-18.01-f07-lec01_300k.srt', 22, 120)