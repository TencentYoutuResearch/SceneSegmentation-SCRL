import pickle
import torch
import torch.utils.data as data
import numpy as np
import random

class MovieNet_SceneSeg_Dataset_Embeddings_Train(data.Dataset):
    def __init__(self, pkl_path, frame_size=3, shot_num=1, 
        sampled_shot_num=10, shuffle_p=0.5,random_cat=False):
        self.shot_num = shot_num
        self.pkl_path = pkl_path
        self.frame_size = frame_size
        self.sampled_shot_num = sampled_shot_num
        self.shuffle_p = shuffle_p
        self.dict_idx_shot = {}
        self.data_length = 0
        self.random_cat = random_cat
        fileObject = open(self.pkl_path, 'rb')
        self.pickle_data = pickle.load(fileObject)
        fileObject.close()
        self.total_video_num = len(self.pickle_data.keys())
        idx = 0
        self.shuffle_map = {}
        self.shuffle_offset = {}
        for k, v in self.pickle_data.items():
            video_shot_group_num = (len(v) // self.sampled_shot_num) - 1
            self.shuffle_map[k] = (len(v) - self.sampled_shot_num * video_shot_group_num)
            self.shuffle_offset[k] = 0
            for i in range(video_shot_group_num):
                self.dict_idx_shot[idx] = (k, i)
                idx += 1
        self._shuffle_offset()
        print(f'Train video num: {self.total_video_num}')
        print(f'total shot group: {idx}')
        self.data_length = idx

    def _shuffle_offset(self):
        for k, offset_upper_bound in self.shuffle_map.items():
            offset = random.randint(0, offset_upper_bound-1)
            offset = 0 if offset < 0 else offset
            self.shuffle_offset[k] = offset
    
    def _get_randomly_cat_clip(self, idx):
        k, i = self.dict_idx_shot[idx]
        sampled_len = self.sampled_shot_num // 2
        # randomly cat an another clip
        data1, label1, _ = self._get_clip_by_idx(idx, sampled_len)
        # fix last shot label
        label1[-1] = 1
        # random the index
        length = len(self.pickle_data[k])
        start = random.randint(0, length - sampled_len - 1)

        p = self.pickle_data[k][start : start + sampled_len]
        data = np.array([p[i][0] for i in range(sampled_len)])
        label = np.array([p[i][1] for i in range(sampled_len)])
        data2 = torch.from_numpy(data).squeeze(1)
        label2 = torch.from_numpy(label).long()

        data = torch.cat([data1, data2],dim=0)
        label = torch.cat([label1, label2],dim=0)
        return data, label, k


    def _seg_shuffle(self, data, label):
        new_d, new_l = [], []
        clips = []
        # find positive pos
        p_index = torch.where(label>=1)[0]
        start, end = 0, len(label)
        for i in p_index:
            i = i.item()
            clips.append((start, i+1))
            start = i+1
        if start != end:
            clips.append((start, end))
            # if the last clip is used for shulling 
            # the label of the last shot might be changed
            label[-1] = 1
        clips_len = len(clips)
        index_list = random.sample(range(0, clips_len), clips_len)
        for i in index_list:
            s, e = clips[i]
            new_d.append(data[s:e])
            new_l.append(label[s:e])
        d = torch.cat(new_d,dim=0)
        l = torch.cat(new_l,dim=0)
        # when shuffling is done, fix the last shot label
        l[-1] = 0
        return d, l

    def _get_clip_by_idx(self, idx, length):
        k , i = self.dict_idx_shot[idx]
        offset = self.shuffle_offset[k]
        s = self.sampled_shot_num
        p = self.pickle_data[k][i*s+offset:(i+1)*s+offset][:length]
        data = np.array([p[i][0] for i in range(length)])
        label = np.array([p[i][1] for i in range(length)])
        data = torch.from_numpy(data).squeeze(1)
        label = torch.from_numpy(label).long()
        # fix last shot label
        label[-1] = 0
        return data, label, k


    def __getitem__(self, idx):
        if not self.random_cat:
            data, label, k = self._get_clip_by_idx(idx, self.sampled_shot_num)
        else:
            data, label, k = self._get_randomly_cat_clip(idx)
        if random.random() < self.shuffle_p:
            data, label = self._seg_shuffle(data, label)
        return data, label, k

    def __len__(self):
        return self.data_length

class MovieNet_SceneSeg_Dataset_Embeddings_Val(data.Dataset):
    def __init__(self, pkl_path, frame_size=3, shot_num=1, 
        sampled_shot_num=100):
        self.shot_num = shot_num
        self.pkl_path = pkl_path
        self.frame_size = frame_size
        self.sampled_shot_num = sampled_shot_num
        self.dict_idx_shot = {}
        self.data_length = 0
        fileObject = open(self.pkl_path, 'rb')
        self.pickle_data = pickle.load(fileObject)
        fileObject.close()
        self.total_video_num = len(self.pickle_data.keys())
        idx = 0
        for k, v in self.pickle_data.items():
            self.dict_idx_shot[idx] = (k, v)
            idx += 1
        print(f'video num: {self.total_video_num}')
        self.data_length = idx

    def _padding(self, data):
        stride = self.sampled_shot_num // 2
        shot_len = data.size(0)
        p_l = data[0].repeat(self.sampled_shot_num // 4, 1)
        p_r_len = self.sampled_shot_num // 4
        res = shot_len % (stride)
        if res != 0:
            p_r_len += (stride) - res
        p_r = data[-1].repeat(p_r_len, 1)
        pad_data = torch.cat((p_l, data, p_r),0)
        assert pad_data.size(0) % stride == 0
        return pad_data

    def __getitem__(self, idx):
        k, v = self.dict_idx_shot[idx]
        num_shot = len(v)
        data = np.array([v[i][0] for i in range(num_shot)])
        label = np.array([v[i][1] for i in range(num_shot)])
        data = torch.from_numpy(data).squeeze(1)
        data = self._padding(data)
        label = torch.from_numpy(label)
        return data, label, k

    def __len__(self):
        return self.data_length