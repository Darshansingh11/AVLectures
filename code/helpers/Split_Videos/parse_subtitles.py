import os
from os.path import join
import argparse
import ast
from glob import glob

class ParseSubtitle():

    def __init__(self, base_dir=None, min_time=7, max_time=15):
        
        self.base_dir = os.curdir if base_dir is None else base_dir
        self.min_time = min_time
        self.max_time = max_time
        self.delimiter = "@#@"

    def getTime(self, t):
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
    
    def toFFMPEGtime(self, t):
        ss, ms = divmod(t*1000, 1000)
        mm, ss = divmod(ss, 60)
        hh, mm = divmod(mm, 60)

        return "{:02d}:{:02d}:{:02d}.{:03d}".format(int(hh), int(mm), int(ss), int(ms))
    
    def parse(self, filename):
        
        filename_split = filename.split('/')
        base_dir, filename = "/".join(filename_split[:-1]), filename_split[-1]

        name = filename.replace('.srt', '')

        outfile = join(base_dir, name + "_parsed.txt")

        sub_file = open(outfile, 'w')
        self.lines = []
        with open(str(join(base_dir, filename)), 'r') as f:
            lines = f.readlines()
            
            row, st, en = "", "", ""
            start, num = 0, 0
            st_prev = ""
            en_prev = ""
            
            for line in lines:
                l = line.strip()
                l = l.replace(self.delimiter, '')  # newly added
                if "-->" in l:
                    start = 1
                    row = ""
                    tm = l.split(" ")
                    st, en = tm[0].strip(), tm[2].strip()
                elif l != "":
                    row += l + " "
                else:
                    row += self.delimiter + st + self.delimiter + en
                    if start:
                        row = "{}-{:05d}.mp4{}{}\n".format(name, num, self.delimiter, row)
                        self.lines.append(row)
                        sub_file.write(row)
                        num += 1
                        start = 0
                    st_prev = st
                    en_prev = en

            if((st != st_prev or en == en_prev) and (st == en_prev)):
                row += self.delimiter + st + self.delimiter + en
                if start:
                    row = "{}-{:05d}.mp4{}{}\n".format(name, num, self.delimiter, row)
                    self.lines.append(row)
                    sub_file.write(row)
                    num += 1
                    start = 0

        sub_file.close()
        self.merge(name, base_dir)

    def merge(self, yid, base_dir):
        
        outfile = join(base_dir, yid + "_merged.txt")
        sub_file = open(outfile, 'w')

        s = '00:00:00.000'
        make_start = 1
        gtLessmintime = False
        ll = []
        tm = []
        cnt = 0
        for line in self.lines:

            #row = line.strip().split('|')
            row = line.strip().split(self.delimiter)

            vid_id ,start, end = row[0], row[2], row[3]

            vid_id = "-".join(vid_id.split('-')[:-1])

            if make_start:
                s = start

            gt = self.getTime(end) - self.getTime(s)

            if self.min_time <= gt <= self.max_time:
                gtLessmintime = False
                ll.append(row[1])
                tm.append([row[1], row[2], row[3]])
                sen = " ".join(ll)

                sub_file.write('{1}-{2:05d}.mp4{0}{3}{0}{4}{0}{5}\n'.format(self.delimiter, vid_id, cnt, sen, tm[0][1], tm[-1][2]))
                cnt += 1
                make_start = 1
                ll = []
                tm = []
            elif gt < self.min_time:
                gtLessmintime = True

                make_start = 0
                ll.append(row[1])
                tm.append([row[1], row[2], row[3]])
            
            else: # if gt > self.max_time
                gtLessmintime = False
                ll.append(row[1])
                tm.append([row[1], row[2], row[3]])
                sen = " ".join(ll)

                sub_file.write('{1}-{2:05d}.mp4{0}{3}{0}{4}{0}{5}\n'.format(self.delimiter, vid_id, cnt, sen, tm[0][1], tm[-1][2]))
                cnt += 1
                make_start = 1
                ll = []
                tm = []

        if gtLessmintime:
            sen = " ".join(ll)
            sub_file.write('{1}-{2:05d}.mp4{0}{3}{0}{4}{0}{5}\n'.format(self.delimiter, vid_id, cnt, sen, tm[0][1], tm[-1][2]))

        sub_file.close()

    def combine(self):

        combined_file = open(join(self.base_dir, 'combined.txt'), 'w')
        for fl in glob(join(self.base_dir, 'subtitles', '*_merged.txt')):

            with open(fl, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    combined_file.write(line)
            
        combined_file.close()