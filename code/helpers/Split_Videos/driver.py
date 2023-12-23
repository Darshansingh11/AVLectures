import argparse
import subprocess
import os

from os.path import join
from glob import glob

import make_splits as ms
import parse_subtitles as ps

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", type=str, required=False, help="Path of DataSubset", default = '/ssd_scratch/cvit/AVLectures/DataSubset')
parser.add_argument("-x", "--min_time", type=int, required=False, help="Minimum time (in seconds)", default = 7)
parser.add_argument("-y", "--max_time", type=int, required=False, help="Maximum time (in seconds)", default = 15)

args = parser.parse_args()

base_dir = args.base_dir
min_time = args.min_time
max_time = args.max_time

print("Base Directory: ", base_dir)
print("Min time = ", min_time, "seconds")
print("Max time = ", max_time, "seconds")

dir_list = []

for dir in glob(join(base_dir, '*')):
     if 'mit' in dir:
          dir_list.append(dir)

dir_list.sort()
print(dir_list)

for dir in dir_list:

     print("Inside Directory - ", dir)

     base_dir = dir

     # step 1

     p = ps.ParseSubtitle(base_dir = base_dir, min_time = min_time, max_time = max_time)

     srt_files = []

     for fl in glob(join(base_dir, 'subtitles', '*')):
          if fl.endswith('.srt'):
               srt_files.append(fl)
     
     for fl in srt_files:
          print(fl)
          p.parse(fl)
     
     # step 2

     p.combine()

     # step 3
     
     m = ms.SplitVideo(base_dir = base_dir)
     m._split()

print("Done Successfully")
