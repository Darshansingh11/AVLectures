from finch import FINCH
import numpy as np
import os
import cv2
import sys
import pickle as pkl
from tqdm import tqdm

def check_clusters(cluster):
    prev_cluster = cluster[0]
    for c in cluster:
        if (c == prev_cluster) or (c == prev_cluster + 1):
            prev_cluster = c
        else:
            return False
    
    return True

# learned lecture aware embeddings on which you want to perform clustering using TW-FINCH 
a = pkl.load(open('/ssd_scratch/cvit/darshan_2/seg_embds/2d3dOCR_ss_test50ft50.pkl', 'rb'))
b = pkl.load(open('/home2/darshan.singh/combined_stats.pkl', 'rb'))

d = {}
cnt = 0
cnt_2, cnt_3, cnt_4 = 0, 0, 0

for course in a:
    course_cnt = 0
    for lec in a[course]:
        course_cnt += 1
    print("For course", course, "->", course_cnt)

lst_alpha = []

for course in tqdm(list(a.keys())):
    d[course] = {}
    for lec in a[course]:
        cnt += 1
        clusters = b[course][lec]['num_segments']
        vid_emb = a[course][lec]['vid_embd']
        txt_emb = a[course][lec]['text_embd']
        vidtext_emb = np.hstack((vid_emb, txt_emb))
        d[course][lec] = {}    

        _, _, vid_clusters = FINCH(vid_emb, req_clust=clusters, tw_finch=True, alpha = 1)
        alpha = 1
        while not check_clusters(vid_clusters):
            _, _, vid_clusters = FINCH(vid_emb, req_clust=clusters, tw_finch=True, alpha = alpha)
            alpha += 0.1
        if alpha > 5:
            lst_alpha.append(alpha)

        if not check_clusters(vid_clusters):
            cnt_2 += 1

        _, _, txt_clusters = FINCH(txt_emb, req_clust=clusters, tw_finch=True, alpha = 1)
        loop_counter = 0
        alpha = 1
        while not check_clusters(txt_clusters):
            _, _, txt_clusters = FINCH(txt_emb, req_clust=clusters, tw_finch=True, alpha = alpha)
            alpha += 0.1
            loop_counter += 1
        if alpha > 5:
            lst_alpha.append(alpha)

        if not check_clusters(txt_clusters):
            cnt_3 += 1

        _, _, vidtxt_clusters = FINCH(vidtext_emb, req_clust=clusters, tw_finch=True, alpha = 1)

        alpha = 1
        while not check_clusters(vidtxt_clusters):
            _, _, vidtxt_clusters = FINCH(vidtext_emb, req_clust=clusters, tw_finch=True, alpha = alpha)
            alpha += 0.1
        if alpha > 5:
            lst_alpha.append(alpha)

        if not check_clusters(vidtxt_clusters):
            cnt_4 += 1

        vid_clusters = vid_clusters.reshape(-1)
        txt_clusters = txt_clusters.reshape(-1)
        vidtxt_clusters = vidtxt_clusters.reshape(-1)
        d[course][lec]['vid_clusters'] = vid_clusters
        d[course][lec]['txt_clusters'] = txt_clusters
        d[course][lec]['vidtxt_clusters'] = vidtxt_clusters

print(cnt, len(lst_alpha), cnt_2, cnt_3, cnt_4)
print(lst_alpha)

clusters = d.copy()
lecs_excluded = {}

for course in clusters:
    for lec in clusters[course]:
        for t in clusters[course][lec]:
            lst = []
            for i in clusters[course][lec][t]:
                lst.append(i)
            prev_cluster = lst[0]
            for i, c in enumerate(lst):
                if (c == prev_cluster) or (c == prev_cluster + 1):
                    prev_cluster = c
                else:
                    if course not in lecs_excluded:
                        lecs_excluded[course] = []
                    if lec not in lecs_excluded[course]:
                        lecs_excluded[course].append(lec)
                    break

for course in lecs_excluded:
    for lec in lecs_excluded[course]:
        print(course, '->', lec)

from os.path import join

# Path where you want to save the predicted clusters
with open(join('/ssd_scratch/cvit/darshan_2/clusters/2d3dOCR_ss_test50ft50.pkl'), 'wb') as handle:
    pkl.dump(d, handle, protocol=pkl.HIGHEST_PROTOCOL)