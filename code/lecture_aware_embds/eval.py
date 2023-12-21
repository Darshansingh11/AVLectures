from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
from args import get_args
from model import Net
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
    we = None

if args.eval_avlectures:
    dataset_avlectures = AVLectures_DataLoader(
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
    dataloader_avlectures = DataLoader(
        dataset_avlectures,
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

def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating Text-Video retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            text = data['text'].cuda()
            video = data['video'].cuda()
            vid = data['video_id']
            ocr_embd = None
            if args.ocr:
                ocr_embd = data['ocr_embd'].cuda()
            m = model(video, text, ocr_embd)
            m  = m.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)

all_checkpoints = glob.glob(args.pretrain_path)

for c in all_checkpoints:
    print('Eval checkpoint: {}'.format(c))
    print('Loading checkpoint: {}'.format(c))
    net.load_checkpoint(c)
    if args.eval_avlectures:
        Eval_retrieval(net, dataloader_avlectures, 'AVLectures')
