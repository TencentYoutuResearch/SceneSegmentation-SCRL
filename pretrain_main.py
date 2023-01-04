import argparse
import yaml
import os, shutil
import builtins
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import random
import numpy as np

from models import factory
from utils import to_log, set_log
from pretrain_trainer import train_SCRL


def start_training(cfg):
    # only multiprocessing_distributed is supported
    if cfg['DDP']['multiprocessing_distributed']:
        
        ngpus_per_node = torch.cuda.device_count()

        if cfg['DDP']['dist_url'] == "env://":
            os.environ['MASTER_ADDR'] = cfg['DDP']['master_ip']
            os.environ['MASTER_PORT'] = str(cfg['DDP']['master_port'])
            os.environ['WORLD_SIZE'] = str(ngpus_per_node * cfg['DDP']['machine_num'])
            os.environ['NODE_RANK'] = str(cfg['DDP']['node_num'])
            os.environ['NUM_NODES'] = str(cfg['DDP']['machine_num'])
            os.environ['NUM_GPUS_PER_NODE'] = str(ngpus_per_node)
            # os.environ['NCCL_IB_DISABLE'] = "1"

        cfg['DDP']['world_size'] = ngpus_per_node * cfg['DDP']['machine_num']
        print(cfg['DDP']['world_size'], ngpus_per_node)

        mp.spawn(task_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    

def setup_worker(seed, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.set_device(gpu)

def task_worker(gpu, ngpus_per_node, cfg):
    setup_worker(seed = 100, gpu = gpu)
    if gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    cfg['DDP']['rank']= cfg['DDP']['node_num'] * ngpus_per_node + gpu
    cfg['DDP']['gpu'] = gpu 
    if cfg['DDP']['dist_url'] == 'env://':
        os.environ['RANK'] = str(cfg['DDP']['rank'])
        
    print(cfg['DDP']['dist_backend'], cfg['DDP']['dist_url'], cfg['DDP']['world_size'],cfg['DDP']['rank'] )
    dist.init_process_group(backend=cfg['DDP']['dist_backend'], init_method=cfg['DDP']['dist_url'])
    
    if gpu == 0:
        to_log(cfg, 'DDP init succeed!', True)

    model, train_loader, train_sampler, criterion, optimizer \
        = factory.get_training_stuff(cfg, gpu, ngpus_per_node)
    
    # training function
    if cfg['model']['SSL'] == 'SCRL':
        train_fun = train_SCRL
    else:
        raise NotImplementedError
    
    start_epoch = cfg['optim']['start_epoch']
    end_epoch = cfg['optim']['epochs']

    assert train_fun is not None
    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, cfg['optim']['lr'], epoch, cfg)
        train_fun(gpu, train_loader, model, criterion, optimizer, epoch, cfg)
        if cfg['DDP']['rank'] == 0 and (epoch + 1) % 4 == 0:
            save_checkpoint(cfg,{
                'epoch': epoch + 1,
                'arch': cfg['model']['backbone'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def adjust_learning_rate(optimizer, init_lr, epoch, cfg):
    """Decay the learning rate based on schedule"""
    if cfg['optim']['lr_cos'] == True:
        cur_lr = init_lr * 0.5 * (1. + math.cos(0.5 * math.pi * epoch / cfg['optim']['epochs']))
    else:
        cur_lr = init_lr
        for milestone in cfg['optim']['schedule']:
            cur_lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr



def save_checkpoint(cfg, state, is_best, filename='checkpoint.pth.tar'):
    p = os.path.join(cfg['log']['dir'], 'checkpoints')
    if not os.path.exists(p):
        os.makedirs(p)
    
    torch.save(state, os.path.join(p, filename))
    if is_best:
        shutil.copyfile(os.path.join(p, filename), os.path.join(p, 'model_best.pth.tar'))

    
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/SCRL_pretrain_default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf8'))
    cfg = set_log(cfg)
    shutil.copy(args.config, cfg['log']['dir'])
    return cfg


def main():
    cfg = get_config()
    start_training(cfg)

if __name__ == '__main__':
    main()
