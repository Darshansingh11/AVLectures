from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
from collections import defaultdict
from torch.utils.data.dataloader import default_collate

from stop_words import ENGLISH_STOP_WORDS


class AVLectures_DataLoader(Dataset):
    """AVLectures dataset loader."""

    def __init__(
            self,
            data,
            helper_pkl,
            we,
            max_words=30,
            we_dim=300,
            word2vec=True,
            ocr=0,
            n_pair=1,
            only_2d=0,
            only_3d=0
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb'))
        if n_pair > 1:
            self.helper_pkl = pickle.load(open(helper_pkl, 'rb'))
        # self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.word2vec= word2vec
        self.ocr = ocr
        if word2vec:
            self.we = we
        self.n_pair = n_pair
        self.only_2d = only_2d
        self.only_3d = only_3d
        # else:
        #     self.we = pickle.load(open(we, 'rb'))

    def __len__(self):
        return len(self.data)

    # def custom_collate(self, batch):
    #     return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        # words = [word for word in words if word in self.we.vocab]
        words = list(map(lambda word: word.lower(), words))
        words = [word for word in words if (word in self.we.vocab) and (word not in ENGLISH_STOP_WORDS)]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)
    
    # orig
    # def r_sampling(self, vid_id, n_pair):
    #     from random import choices
    #     import copy
    #     vid_name = vid_id.split('-')[0]
    #     vid_list = copy.deepcopy(self.helper_pkl[vid_name])
    #     # for i in self.data:
    #     #     vi_name = i['id'].split('-')[0]
    #     #     if vi_name == vid_name:
    #     #         vid_list.append(i)

    #     r_lst = choices(vid_list, k=n_pair)
    #     vid_feats = None
    #     capts = None
    #     ocr_cat = None
    #     for r in r_lst:
    #         feat_2d = F.normalize(th.from_numpy(r['2d']).float(), dim=0)
    #         feat_3d = F.normalize(th.from_numpy(r['3d']).float(), dim=0)
    #         cap = r['caption']
    #         caption = self._words_to_we(self._tokenize_text(cap))
    #         caption = caption.unsqueeze(0)

    #         ocr_text = r['ocr_text']
    #         ocr_embd = self._words_to_we(self._tokenize_text(ocr_text))
    #         ocr_embd = ocr_embd.unsqueeze(0)


    #         # print(th.cat((feat_2d, feat_3d)).shape, caption.shape)
    #         if vid_feats is None:
    #             vid_feats = th.cat((feat_2d, feat_3d)).unsqueeze(0)
    #         else:
    #             t = th.cat((feat_2d, feat_3d))
    #             t = t.unsqueeze(0)
    #             vid_feats = th.cat((vid_feats, t))

    #         if capts is None:
    #             capts = caption
    #         else:
    #             capts = th.cat((capts, caption))

    #         if ocr_cat is None:
    #             ocr_cat = ocr_embd
    #         else:
    #             ocr_cat = th.cat((ocr_cat, ocr_embd))
        
    #     return capts, vid_feats, ocr_cat


    # new

    def r_sampling(self, vid_name, n_pair):
        from random import choices
        import copy
        vid_list = copy.deepcopy(self.helper_pkl[vid_name])
        # for i in self.data:
        #     vi_name = i['id'].split('-')[0]
        #     if vi_name == vid_name:
        #         vid_list.append(i)

        r_lst = choices(vid_list, k=n_pair)
        # r_lst = list(np.random.choice(vid_list, n_pair, replace=False))
        
        vid_feats = None
        capts = None
        ocr_cat = None
        for r in r_lst:
            feat_2d = F.normalize(th.from_numpy(r['2d']).float(), dim=0)
            feat_3d = F.normalize(th.from_numpy(r['3d']).float(), dim=0)

            if self.word2vec:
                cap = r['caption']
                caption = self._words_to_we(self._tokenize_text(cap))
                caption = caption.unsqueeze(0)

                if 'ocr_text' in r:
                    ocr_text = r['ocr_text']
                    ocr_embd = self._words_to_we(self._tokenize_text(ocr_text))
                    ocr_embd = ocr_embd.unsqueeze(0)

            else:
                caption = th.from_numpy(r['emb_ss'])
                caption = caption.unsqueeze(0)

                if 'ocr_text' in r:
                    ocr_embd = th.from_numpy(r['ocr_emb_ss'])
                    ocr_embd = ocr_embd.unsqueeze(0)
                    # print(ocr_embd.shape)


            # print(th.cat((feat_2d, feat_3d)).shape, caption.shape)
            if vid_feats is None:
                if self.only_2d:
                    vid_feats = feat_2d.unsqueeze(0)
                elif self.only_3d:
                    vid_feats = feat_3d.unsqueeze(0)
                else:
                    vid_feats = th.cat((feat_2d, feat_3d)).unsqueeze(0)
            else:
                if self.only_2d:
                    t = feat_2d.unsqueeze(0)
                elif self.only_3d:
                    t = feat_3d.unsqueeze(0)
                else:
                    t = th.cat((feat_2d, feat_3d)).unsqueeze(0)
                # t = th.cat((feat_2d, feat_3d))
                # t = t.unsqueeze(0)
                vid_feats = th.cat((vid_feats, t))

            if capts is None:
                capts = caption
            else:
                capts = th.cat((capts, caption))

            if 'ocr_text' in r:
                if ocr_cat is None:
                    ocr_cat = ocr_embd
                else:
                    ocr_cat = th.cat((ocr_cat, ocr_embd))
        
        # print(capts.shape, ocr_cat.shape)
        if not self.word2vec:
            capts = capts.unsqueeze(1)
            ocr_cat = ocr_cat.unsqueeze(1)
        
        return capts, vid_feats, ocr_cat



    # orig
    # def __getitem__(self, idx):
    #     feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
    #     feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
    #     video = th.cat((feat_2d, feat_3d))

    #     # feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
    #     # video = feat_2d

    #     # feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
    #     # video = feat_3d



    #     if self.word2vec:
    #         cap = self.data[idx]['caption']
    #         caption = self._words_to_we(self._tokenize_text(cap))
    #         # print("CAPTION SHAPE", caption.shape)
    #         if self.ocr:
    #             ocr_text = self.data[idx]['ocr_text']
    #             ocr_embd = self._words_to_we(self._tokenize_text(ocr_text))
        
    #     else:
    #         #print(self.we[self.data[idx]['id']].keys())
    #         caption = self.data[idx]['emb_ss']
    #         if self.ocr:
    #             ocr_embd = self.data[idx]['ocr_emb_ss']
        
    #     # a, b = self.r_sampling(self.data[idx]['id'], 64)
    #     # print("a:", a.shape, "b", b.shape)
    #     # print("vid:", video.shape, "cap", caption.shape)

    #     caption, video, ocr_embd = self.r_sampling(self.data[idx]['id'], self.n_pair)

    #     if self.ocr:
    #         return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'], 
    #         'course_name': self.data[idx]['course_name'], 'st': self.data[idx]['st'], 'et': self.data[idx]['et'], 
    #         'vid_duration': self.data[idx]['vid_duration'], 'ocr_embd': ocr_embd}
    #     else:
    #         return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'], 
    #         'course_name': self.data[idx]['course_name'], 'st': self.data[idx]['st'], 'et': self.data[idx]['et'], 
    #         'vid_duration': self.data[idx]['vid_duration']}

    #     # return {'video': video, 'text': caption, 'video_id': self.data[idx]['id']}




# new

    def __getitem__(self, idx):

        if self.n_pair > 1:
            caption, video, ocr_embd = self.r_sampling(self.data[idx], self.n_pair)
            # print(caption.shape, video.shape)
        else:
            if self.only_2d:
                feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
                video = feat_2d
            elif self.only_3d:
                feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
                video = feat_3d
            else:
                feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
                feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
                video = th.cat((feat_2d, feat_3d))
            

            # feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
            # video = feat_2d

            # feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
            # video = feat_3d

            if self.word2vec:
                cap = self.data[idx]['caption']
                caption = self._words_to_we(self._tokenize_text(cap))
                if self.ocr:
                    ocr_text = self.data[idx]['ocr_text']
                    ocr_embd = self._words_to_we(self._tokenize_text(ocr_text))
            
            else:
                caption = self.data[idx]['emb_ss']
                if self.ocr:
                    ocr_embd = self.data[idx]['ocr_emb_ss']
        
        if self.n_pair > 1:
            if self.ocr:
                return {'video': video, 'text': caption, 'video_id': self.data[idx], 'ocr_embd': ocr_embd}
            else:
                return {'video': video, 'text': caption, 'video_id': self.data[idx]}
        
        else:

            if self.ocr:
                return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'], 
                'course_name': self.data[idx]['course_name'], 'st': self.data[idx]['st'], 'et': self.data[idx]['et'], 
                'vid_duration': self.data[idx]['vid_duration'], 'ocr_embd': ocr_embd}
            else:
                return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'], 
                'course_name': self.data[idx]['course_name'], 'st': self.data[idx]['st'], 'et': self.data[idx]['et'], 
                'vid_duration': self.data[idx]['vid_duration']}



        # return {'video': video, 'text': caption, 'video_id': self.data[idx]['id']}
