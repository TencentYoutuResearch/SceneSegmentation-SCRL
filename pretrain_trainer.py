import time
import torch
import torch.nn.parallel
import torch.optim
from utils import AverageMeter, ProgressMeter, to_log, accuracy



def train_SCRL(gpu, train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    gradient_clip_val = cfg['optim']['gradient_norm']

    model.train()
    view_size = (-1, 3 * cfg['data']['frame_size'], 224, 224)
    pivot = time.time()
    for i, data in enumerate(train_loader):
        if gpu is not None:
            data_q = data[0].cuda(gpu, non_blocking=True)
            data_k = data[1].cuda(gpu, non_blocking=True)
        data_time.update(time.time() - pivot)
        data_q = data_q.view(view_size)
        data_k = data_k.view(view_size)

        output, target = model(data_q, data_k)
        
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        optimizer.step()

        batch_time.update(time.time() - pivot)
        pivot = time.time()

        if gpu == 0 and i % cfg['log']['print_freq'] == 0:
            _out = progress.display(i)
            to_log(cfg, _out, True)
