from PIL import ImageFilter
import random
import torch
import torchvision.transforms as transforms
import json 
import cv2
import numpy as np
from torchvision import utils as vutils

class TwoWayTransform:
    def __init__(self, base_transform_a,
        base_transform_b, fixed_aug_shot=True):
        self.base_transform_a = base_transform_a
        self.base_transform_b = base_transform_b
        self.fixed = fixed_aug_shot

    def __call__(self, x):
        frame_num = len(x)
        if self.fixed:
            seed = np.random.randint(2147483647)
            q, k = [], []
            for i in range(frame_num):
                random.seed(seed)
                q.append(self.base_transform_a(x[i]))
            seed = np.random.randint(2147483647)
            for i in range(frame_num):
                random.seed(seed)
                k.append(self.base_transform_b(x[i]))
        else:
            q = [self.base_transform_a(x[i]) for i in range(frame_num)]
            k = [self.base_transform_b(x[i]) for i in range(frame_num)]
        q = torch.cat(q, axis = 0)
        k = torch.cat(k, axis = 0)
        return [q, k]


class MovieNet_Shot_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, shot_info_path, transform,
        shot_len = 16, frame_per_shot = 3, _Type='train'):
        self.img_path = img_path
        with open(shot_info_path, 'rb') as f:
            self.shot_info = json.load(f)
        self.img_path = img_path
        self.shot_len = shot_len
        self.frame_per_shot = frame_per_shot
        self.transform = transform
        self._Type = _Type.lower()
        assert self._Type in ['train','val','test']
        self.idx_imdb_map = {}
        data_length = 0
        for imdb, shot_num in self.shot_info[_Type].items():
            for i in range(shot_num // shot_len):
                self.idx_imdb_map[data_length] = (imdb, i)
                data_length += 1

            
    def __len__(self):
        return len(self.idx_imdb_map.keys())


    def _transform(self, img_list):
        q, k = [], []
        for item in img_list:
            out = self.transform(item)
            q.append(out[0])
            k.append(out[1])
        out_q = torch.stack(q, axis=0)
        out_k = torch.stack(k, axis=0)
        return [out_q, out_k]


    def _process_puzzle(self, idx):
        imdb, puzzle_id = self.idx_imdb_map[idx]
        img_path =  f'{self.img_path}/{imdb}/{str(puzzle_id).zfill(4)}.jpg'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.vsplit(img, self.shot_len)
        img = [np.hsplit(i, self.frame_per_shot) for i in img]
        data = self._transform(img)
        return data


    def __getitem__(self, idx):
        return self._process_puzzle(idx)



class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_train_loader(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
    augmentation_base = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    augmentation_color = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    augmentation_q = augmentation_color if cfg['data']['color_aug_for_q'] else augmentation_base
    augmentation_k = augmentation_color if cfg['data']['color_aug_for_k'] else augmentation_base

    train_transform = TwoWayTransform(
        transforms.Compose(augmentation_q), 
        transforms.Compose(augmentation_k),
        fixed_aug_shot=cfg['data']['fixed_aug_shot'])
    
    img_path = cfg['data']['data_path'] 
    shot_info_path = cfg['data']['shot_info'] 
    train_dataset = MovieNet_Shot_Dataset(img_path, shot_info_path, train_transform)
    train_sampler = None
    if cfg['DDP']['multiprocessing_distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=cfg['optim']['bs'], num_workers=cfg['data']['workers'],
        sampler=train_sampler, shuffle=(train_sampler is None), pin_memory=True, drop_last=True)
    return train_loader, train_sampler

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
    augmentation_base = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]
    augmentation_color = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]
    train_transform = TwoWayTransform(
        transforms.Compose(augmentation_base), 
        transforms.Compose(augmentation_color),
        fixed_aug_shot=False)
    img_path = './compressed_shot_images'
    shot_info_path = './MovieNet_shot_num.json'
    train_dataset = MovieNet_Shot_Dataset(img_path, shot_info_path, train_transform)
    print(f'len: {len(train_dataset)}')
    i = train_dataset[0]
    print(i[0].size())






    