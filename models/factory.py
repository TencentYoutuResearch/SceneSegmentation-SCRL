import models.backbones.visual.resnet as resnet
from models.core.SCRL_MoCo import SCRL
from data.movienet_data import get_train_loader
import torch, os
from utils import to_log

def get_model(cfg):
    encoder = None
    model = None
    if 'multimodal' not in cfg or cfg['multimodal']['using_audio'] == False:
        encoder = resnet.encoder_resnet50
    else:
        raise NotImplementedError
    assert encoder is not None
    
    to_log(cfg, 'backbone init: ' + cfg['model']['backbone'], True)

    if cfg['model']['SSL'] == 'SCRL':
            model = SCRL(
            base_encoder                = encoder,
            dim                         = cfg['MoCo']['dim'], 
            K                           = cfg['MoCo']['k'], 
            m                           = cfg['MoCo']['m'], 
            T                           = cfg['MoCo']['t'], 
            mlp                         = cfg['MoCo']['mlp'], 
            encoder_pretrained_path     = cfg['model']['backbone_pretrain'],
            multi_positive              = cfg['MoCo']['multi_positive'],
            positive_selection          = cfg['model']['Positive_Selection'],
            cluster_num                 = cfg['model']['cluster_num'],
            soft_gamma                  = cfg['model']['soft_gamma'],
            )
    else:
        raise NotImplementedError
    to_log(cfg, 'model init: ' + cfg['model']['SSL'], True)

    if cfg['model']['SyncBatchNorm']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    to_log(cfg, 'SyncBatchNorm: on' if cfg['model']['SyncBatchNorm'] else 'SyncBatchNorm: off', True)
    return model

def get_loader(cfg):
    train_loader, train_sampler = get_train_loader(cfg)
    return train_loader, train_sampler


def get_criterion(cfg):
    criterion = None
    if cfg['model']['SSL'] == 'simsiam':
        criterion = torch.nn.CosineSimilarity(dim=1)
    elif cfg['model']['SSL'] == 'SCRL':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    to_log(cfg, 'criterion init: ' + str(criterion), True)
    return criterion

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['optim']['optimizer'] == 'sgd':
        if cfg['model']['SSL'] == 'simsiam':
            if cfg['model']['fix_pred_lr']:
                optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True}]
            else:
                optim_params = model.parameters()
        elif cfg['model']['SSL'] == 'SCRL':
            optim_params = model.parameters()
        else:
            raise NotImplementedError
        
        optimizer = torch.optim.SGD(optim_params, cfg['optim']['lr'],
                                    momentum=cfg['optim']['momentum'],
                                    weight_decay=cfg['optim']['wd'])
    else:
        raise NotImplementedError
    return optimizer

def get_training_stuff(cfg, gpu, ngpus_per_node):
    cfg['optim']['bs'] = int(cfg['optim']['bs'] / ngpus_per_node)
    to_log(cfg, 'shot per GPU: ' + str(cfg['optim']['bs']), True)

    if cfg['data']['clipshuffle']:
        len_per_data = cfg['data']['clipshuffle_len']
    else:
        len_per_data = 1
    assert cfg['optim']['bs'] % len_per_data == 0
    cfg['optim']['bs'] = int(cfg['optim']['bs'] / len_per_data )
    cfg['data']['workers'] = int(( cfg['data']['workers'] + ngpus_per_node - 1) / ngpus_per_node)
    to_log(cfg, 'batch size per GPU: ' + str(cfg['optim']['bs']), True)
    to_log(cfg, 'worker per GPU: ' +  str(cfg['data']['workers']) , True)

    train_loader, train_sampler = get_train_loader(cfg)
    model = get_model(cfg)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, 
        device_ids=[gpu], 
        output_device=gpu, 
        find_unused_parameters=True)
    
    criterion = get_criterion(cfg).cuda(gpu)
    optimizer = get_optimizer(cfg, model)
    cfg['optim']['start_epoch'] = 0
    resume = cfg['model']['resume']
    if resume is not None and len(resume) > 1:
        if os.path.isfile(resume):
            to_log(cfg, "=> loading checkpoint '{}'".format(resume), True)
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                loc = f'cuda:{gpu}'
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            cfg['optim']['start_epoch'] = start_epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            to_log(cfg, "=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']), True)
        else:
            to_log(cfg, "=> no checkpoint found at '{}'".format(resume), True)
            raise FileNotFoundError
         

    assert model is not None \
        and train_loader is not None \
        and criterion is not None \
        and optimizer is not None
    
    return (model, train_loader, train_sampler, criterion, optimizer)
