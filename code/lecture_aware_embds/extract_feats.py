from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from operator import le

import torch as th
import numpy as np
from torch.utils.data import DataLoader
from args import get_args
from model_ef import Net
# from model import Net
from metrics import compute_metrics, print_computed_metrics
from gensim.models.keyedvectors import KeyedVectors
import pickle
import glob
from lsmdc_dataloader import LSMDC_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from youcook_dataloader import Youcook_DataLoader
from avlectures_dataloader import AVLectures_DataLoader


args = get_args()
if args.verbose:
    print(args)

assert args.pretrain_path != '', 'Need to specify pretrain_path argument'

if args.word2vec:
    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    print('done')
else:
    we = args.BERT_val_path


if args.eval_avlectures:
    dataset_val = AVLectures_DataLoader(
        data=args.avlectures_val_path,
        helper_pkl = args.avlectures_helper_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        word2vec=args.word2vec,
        ocr=args.ocr,
        only_2d=args.only_2d,
        only_3d=args.only_3d
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )


net = Net(
    video_dim=args.feature_dim,
    embd_dim=args.embd_dim,
    we_dim=args.we_dim,
    max_words=args.max_words,
    word2vec=args.word2vec,
    ocr=args.ocr,
    ocr_dim=args.ocr_dim,
    only_ocr=args.only_ocr
)
net.eval()
net.cuda()

if args.verbose:
    print('Starting evaluation loop ...')

pkl_data = {}

def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda()
            video = data['video'].cuda()
            ocr_embd = None
            if args.ocr:
                ocr_embd = data['ocr_embd'].cuda()
            vid = data['video_id']
            course_name = data['course_name'][0]
            st = data['st']
            et = data['et']
            vid_duration = data['vid_duration']
            st = st.item()
            et = et.item()
            vid_duration = vid_duration.item()
            m, vid_embd, text_embd = model(video, text, ocr_embd)

            # for trained embds
            text_embd = text_embd.cpu().detach().numpy()
            vid_embd = vid_embd.cpu().detach().numpy()
            m  = m.cpu().detach().numpy()

            lecture_name = "-".join(vid[0].split('-')[:-1])
            split_number = int(vid[0].split('-')[-1])
            
            if course_name not in pkl_data:
                pkl_data[course_name] = {}
            
            if lecture_name not in pkl_data[course_name]:
                pkl_data[course_name][lecture_name] = {"vid_embd": [], "text_embd": [], "stet": [], "vid_duration": []}

            pkl_data[course_name][lecture_name]["vid_embd"].append((split_number, vid_embd))
            pkl_data[course_name][lecture_name]["text_embd"].append((split_number, text_embd))
            pkl_data[course_name][lecture_name]["stet"].append((split_number, st, et))
            pkl_data[course_name][lecture_name]["vid_duration"].append((split_number, vid_duration))

all_checkpoints = glob.glob(args.pretrain_path)

for c in all_checkpoints:
    print('Eval checkpoint: {}'.format(c))
    print('Loading checkpoint: {}'.format(c))
    net.load_checkpoint(c)
    if args.eval_avlectures:
        Eval_retrieval(net, dataloader_val, 'AVLectures')

pkl_data_new = {}

for c in pkl_data.keys():

    if c not in pkl_data_new:
        pkl_data_new[c] = {}

    for k in pkl_data[c].keys():

        vid_embd_sorted = pkl_data[c][k]['vid_embd']
        vid_embd_sorted = sorted(vid_embd_sorted, key=lambda x: x[0])

        text_embd_sorted = pkl_data[c][k]['text_embd']
        text_embd_sorted = sorted(text_embd_sorted, key=lambda x: x[0])

        vid_duration_sorted = pkl_data[c][k]['vid_duration']
        vid_duration_sorted = sorted(vid_duration_sorted, key=lambda x: x[0])

        stet_sorted = pkl_data[c][k]['stet']
        stet_sorted = sorted(stet_sorted, key=lambda x: x[0])

        vid_embd = vid_embd_sorted[0][1]
        text_embd = text_embd_sorted[0][1]
        vid_duration = [vid_duration_sorted[0][1]]
        stet = [(stet_sorted[0][0], stet_sorted[0][1], stet_sorted[0][2])]s

        for i in range(1, len(vid_embd_sorted)):
            prev_vid_embd = vid_embd
            prev_text_embd = text_embd

            vid_embd = np.concatenate((prev_vid_embd, vid_embd_sorted[i][1]))
            text_embd = np.concatenate((prev_text_embd, text_embd_sorted[i][1]))

            vid_duration.append(vid_duration_sorted[i][1])

            stet.append((stet_sorted[i][0], stet_sorted[i][1], stet_sorted[i][2]))

        pkl_data_new[c][k] = {"vid_embd": vid_embd, "text_embd": text_embd, "vid_duration": vid_duration, "stet": stet}

# Path where you want to save the extracted features
with open('/ssd_scratch/cvit/darshan_2/seg_embds/2d3dOCR_ss_test50ft50.pkl', 'wb') as handle:
	pickle.dump(pkl_data_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("DONE SUCCESSFULLY")