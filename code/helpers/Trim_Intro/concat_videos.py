import subprocess, traceback
import os
import argparse
import pickle as pkl

from os.path import join
from glob import glob
from pathlib import Path
from tqdm import tqdm
import cv2

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = '/ssd_scratch/cvit/darshan/dataset_MITOCW_v1'

num_workers = 20

delimiter = '@@'

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
    return round(frame_count / fps, 3)

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

def process_cmd(cmd):
    try:
        subprocess.call(cmd, shell=True)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()	

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

course_list = []

for course in glob(join(base_dir, '*')):
     if 'mit' in course:
          course_list.append(course)

course_list.sort()

for course in course_list:
    Path(join(course, 'concatenated_videos')).mkdir(parents=True, exist_ok=True)
    Path(join(course, 'concatenated_subtitles')).mkdir(parents=True, exist_ok=True)

    course_name = course.split('/')[-1]
    print("Inside course - ", course_name)


    cmd_list = []

    with open(join(Path.home(), 'Segmentation', 'segments', 'lecname', course_name + '.txt')) as f:
        lines = f.readlines()

        course_stats = OrderedDict()

        for line in lines:
            l = line.strip()
            segment_name = l.split(delimiter)[0]
            lec_name = l.split(delimiter)[-1]

            if lec_name not in course_stats:
                course_stats[lec_name] = []
            course_stats[lec_name].append(segment_name)
    
    for lec_name in course_stats:

        num_segments = len(course_stats[lec_name])
        
        # Usual concat command
        concat_cmd = "ffmpeg -hide_banner -loglevel error"
        for i in range(num_segments):
            concat_cmd += " -i {}".format(join(course, 'videos', course_stats[lec_name][i]))
        concat_cmd += ' -filter_complex "'
        for i in range(num_segments):
            concat_cmd += "[{0}:v] [{0}:a] ".format(i)
        concat_cmd += 'concat=n={}:v=1:a=1 [v] [a]"'.format(num_segments)
        concat_cmd += ' -map "[v]" -map "[a]" {}'.format(join(course, 'concatenated_videos', lec_name + '.mp4'))


        sub_list = []
        for i in range(num_segments):
            sub_list.append(course_stats[lec_name][i].replace('.mp4', '.srt'))

        concat_subtitles(course, lec_name + '.srt', sub_list)

        cmd_list.append(concat_cmd)

    p = ThreadPoolExecutor(num_workers)

    futures = [p.submit(process_cmd, j) for j in cmd_list]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


print("DONE SUCCESSFULLY")