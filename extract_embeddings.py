import pickle
import os
import torch
import argparse
import time
from models.backbones.visual.resnet import encoder_resnet50
import json
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader

class MovieNet_SingleShot_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, shot_info_path, transform,
        frame_per_shot = 3, _Type='train'):
        self.img_path = img_path
        with open(shot_info_path, 'rb') as f:
            self.shot_info = json.load(f)
        self.img_path = img_path
        self.frame_per_shot = frame_per_shot
        self.transform = transform
        self._Type = _Type.lower()
        assert self._Type in ['train','val','test']
        self.idx_imdb_map = {}
        data_length = 0
        for info in self.shot_info[_Type]:
            imdb = info['name']
            for shot in info['label']:
                self.idx_imdb_map[data_length] = (imdb, shot[0], shot[1])
                data_length += 1

    def __len__(self):
        return len(self.idx_imdb_map.keys())

    def _process(self, idx):
        imdb, _id, label = self.idx_imdb_map[idx]
        img_path_0 =  f'{self.img_path}/{imdb}/shot_{_id}_img_0.jpg'
        img_path_1 =  f'{self.img_path}/{imdb}/shot_{_id}_img_1.jpg'
        img_path_2 =  f'{self.img_path}/{imdb}/shot_{_id}_img_2.jpg'
        img_0 = cv2.cvtColor(cv2.imread(img_path_0), cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
        data_0 = self.transform(img_0)
        data_1 = self.transform(img_1)
        data_2 = self.transform(img_2)
        data = torch.cat([data_0, data_1, data_2], axis=0)
        label = int(label)
        # According to LGSS[1]
        # [1] https://arxiv.org/abs/2004.02678
        if label == -1:
            label = 1
        return data, label, (imdb, _id)


    def __getitem__(self, idx):
        return self._process(idx)

def get_loader(cfg, _Type='train'):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    dataset = MovieNet_SingleShot_Dataset(
        img_path = cfg.shot_img_path,
        shot_info_path = cfg.shot_info_path,
        transform = _transform,
        frame_per_shot = cfg.frame_per_shot,
        _Type=_Type,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.bs,  drop_last=False,
        shuffle=False, num_workers=cfg.worker_num, pin_memory=True
    )
    return loader

def get_encoder(model_name='resnet50', weight_path='', input_channel=9):
    encoder = None
    model_name = model_name.lower()
    if model_name == 'resnet50':
        encoder = encoder_resnet50(weight_path='',input_channel=input_channel)
        model_weight = torch.load(weight_path,map_location=torch.device('cpu'))['state_dict']
        pretrained_dict = {}
        for k, v in model_weight.items():
            # moco loading 
            if k.startswith('module.encoder_k'):
                continue
            if k == 'module.queue' or k == 'module.queue_ptr':
                continue
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                k = k[17:]

            pretrained_dict[k] = v
        encoder.load_state_dict(pretrained_dict, strict = False)
        print(f'loaded from {weight_path}')
    return encoder


@torch.no_grad()
def get_save_embeddings(model, loader, shot_num, filename, log_interval=100):
    # dict
    # key: index, value: [(embeddings, label), ...]
    embeddings = {} 
    model.eval()
    
    print(f'total length of dataset: {len(loader.dataset)}')
    print(f'total length of loader: {len(loader)}')
    
    for batch_idx, (data, target, index) in enumerate(loader):
        if batch_idx % log_interval == 0:
            print(f'processed: {batch_idx}')
        
        data = data.cuda(non_blocking=True) # ([bs, shot_num, 9, 224, 224])
        data = data.view(-1, 9, 224, 224)

        target = target.view(-1).cuda()
        output = model(data, False)   # ([bs * shot_num, 2048])
        for i, key in enumerate(index[0]):
            if key not in embeddings:
                embeddings[key] = []
            t_emb = output[i*shot_num:(i+1)*shot_num].cpu().numpy()
            t_label =  target[i].cpu().numpy()
            embeddings[key].append((t_emb.copy() ,t_label.copy()))
    pickle.dump(embeddings, open(filename, 'wb'))


def extract_features(cfg):
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    save_dir = os.path.join(cfg.save_dir, time_str)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    cfg.log_file = save_dir + '/extraction.log'
    encoder = get_encoder(
        model_name=cfg.model_name,
        weight_path=cfg.model_path,
        input_channel=cfg.frame_per_shot * 3
        ).cuda()
    dataType = [cfg.Type]
    if dataType[0] == 'all':
        dataType = ['train','test','val']
    for _T in dataType:
        to_log(cfg, f'processing: {_T} \n')
        loader = get_loader(cfg, _Type = _T)
        filename = os.path.join(save_dir, _T+'.pkl')
        get_save_embeddings(encoder, 
            loader, 
            cfg.shot_num, 
            filename, 
            log_interval=100
        )
        to_log(cfg, f'{_T} embeddings are saved in {filename}!\n')


def to_log(cfg, content, echo=True):
    with open(cfg.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo: print(content)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--shot_info_path', type=str, 
        default='./data/movie1K.scene_seg_318_name_index_shotnum_label.v1.json')
    parser.add_argument('--shot_img_path', type=str, default='./MovieNet_unzip/240P/')
    parser.add_argument('--Type', type=str, default='train', choices=['train','test','val','all'])
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--frame_per_shot', type=int, default=3)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--worker_num', type=int, default=16)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./embeddings/')
    parser.add_argument('--gpu-id', type=str, default='0')
    cfg = parser.parse_args()

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    return cfg


if __name__ == '__main__':
    cfg = get_config()
    extract_features(cfg)