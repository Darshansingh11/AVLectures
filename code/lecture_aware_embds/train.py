from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from args import get_args
import random
import os
from avlectures_dataloader import AVLectures_DataLoader
from model import Net
from metrics import compute_metrics, print_computed_metrics
from loss import MaxMarginRankingLoss
from loss_mms import MMS_loss
from loss_ce import CE_loss
from loss_milnce import MILNCELoss
from gensim.models.keyedvectors import KeyedVectors
import pickle


args = get_args()
if args.verbose:
    print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

if not(args.avlectures):
    print('Loading captions: {}'.format(args.caption_path))
    caption = pickle.load(open(args.caption_path, 'rb'))
    print('done')

if args.word2vec:

    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    
    if args.avlectures:
        we_train = we 
        we_val = we 
    
    print('done')

else:
    we_train = None
    we_val = None

if args.avlectures:
    dataset = AVLectures_DataLoader(
        data=args.avlectures_train_path,
        helper_pkl = args.avlectures_helper_path,
        we=we_train,
        max_words=args.max_words,
        we_dim=args.we_dim,
        word2vec=args.word2vec,
        ocr=args.ocr,
        n_pair=args.n_pair,
        only_2d=args.only_2d,
        only_3d=args.only_3d
    )
dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
)
if args.eval_avlectures:
    dataset_val = AVLectures_DataLoader(
        data=args.avlectures_val_path,
        helper_pkl = args.avlectures_helper_path,
        we=we_val,
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
    n_pair=args.n_pair,
    max_words=args.max_words,
    sentence_dim=args.sentence_dim,
    word2vec=args.word2vec,
    ocr=args.ocr,
    ocr_dim=args.ocr_dim,
    only_ocr=args.only_ocr
)
net.train()
# Optimizers + Loss

loss_op = MaxMarginRankingLoss(
    margin=args.margin,
    negative_weighting=args.negative_weighting,
    batch_size=args.batch_size,
    n_pair=args.n_pair,
    hard_negative_rate=args.hard_negative_rate,
)

# loss_op = MMS_loss()
# loss_op = CE_loss()
# loss_op = MILNCELoss()

net.cuda()
loss_op.cuda()


if args.pretrain_path != '':
    net.load_checkpoint(args.pretrain_path)

optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun):
    text = data['text'].cuda()
    video = data['video'].cuda()
    ocr_embd = None
    if args.ocr:
        ocr_embd = data['ocr_embd'].cuda()
    video = video.view(-1, video.shape[-1])
    if args.word2vec:
        text = text.view(-1, text.shape[-2], text.shape[-1]) # original
        if args.ocr:
            ocr_embd = ocr_embd.view(-1, ocr_embd.shape[-2], ocr_embd.shape[-1])
    else:
        if args.n_pair > 1:
            text = text.view(-1, text.shape[-2], text.shape[-1]) # original
            text = text.squeeze()
            if args.ocr:
                ocr_embd = ocr_embd.view(-1, ocr_embd.shape[-2], ocr_embd.shape[-1])
                ocr_embd =  ocr_embd.squeeze()
    opt.zero_grad()
    with th.set_grad_enabled(True):
        sim_matrix, v, t = model(video, text, ocr_embd)
        loss = loss_fun(sim_matrix)
    loss.backward()
    opt.step()
    return loss.item()

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
            m = model(video, text, ocr_embd)
            m  = m.cpu().detach().numpy()
            metrics = compute_metrics(m)
            print_computed_metrics(metrics)

for epoch in range(args.epochs):
    running_loss = 0.0
    if args.eval_avlectures:
        Eval_retrieval(net, dataloader_val, 'AVLectures')    
    if args.verbose:
        print('Epoch: %d' % epoch)
    for i_batch, sample_batch in enumerate(dataloader):
        batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op) # orig
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose:
            print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
            (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
            running_loss / args.n_display))
            running_loss = 0.0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '':
        if epoch + 1 == args.epochs or (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(epoch + 1))
            net.save_checkpoint(path)

if args.eval_avlectures:
    Eval_retrieval(net, dataloader_val, 'AVLectures')
