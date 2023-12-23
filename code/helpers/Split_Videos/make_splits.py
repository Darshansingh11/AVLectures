from multiprocessing.connection import wait
import subprocess, traceback
import time
import os
from os.path import join
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from glob import glob
import threading

import signal

class SplitVideo():

    # c = 0

    def __init__(self, base_dir):
        
        self.base_dir = base_dir
        self.delimiter = "@#@"

    def process_cmd(self, cmd):
        try:
            subprocess.call(cmd, shell=True)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()		

    def _split(self):

        os.makedirs(join(self.base_dir, 'splits_vid'), exist_ok=True)

        CMD_ffmpeg = []
        with open(join(self.base_dir, 'combined.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                name, _ , start, end = l.strip().split(self.delimiter)
                
                start = start.replace(',', '.')
                end = end.replace(',', '.')
                

                video_file = ""
                videofile_name = join(self.base_dir, "splits_vid", name)

                name = '-'.join(name.split('-')[:-1])

                video_file = join(self.base_dir, 'videos', name + '.mp4')

                cmd1 = 'ffmpeg -hide_banner -loglevel error -ss {} -to {} -i {} -strict -2 {} -y'.format(
                    start, end, video_file, videofile_name
                )

                CMD_ffmpeg.append(cmd1)

        p = ThreadPoolExecutor(20)
        
        futures = [p.submit(self.process_cmd, j) for j in CMD_ffmpeg]
        _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]