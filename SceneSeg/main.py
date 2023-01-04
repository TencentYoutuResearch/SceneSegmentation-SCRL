import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import os
import argparse
import random
import time
from sklearn.metrics import average_precision_score
import shutil
import os.path as osp
from BiLSTM_protocol import BiLSTM
from movienet_seg_data import MovieNet_SceneSeg_Dataset_Embeddings_Train, MovieNet_SceneSeg_Dataset_Embeddings_Val

def main(args):
    setup_seed(100)
    model = BiLSTM(
        input_feature_dim=args.dim,
        input_drop_rate=args.input_drop_rate
    ).cuda()

    label_weights = torch.Tensor([args.loss_weight[0], args.loss_weight[1]]).cuda()
    criterion = nn.CrossEntropyLoss(label_weights).cuda()

    
    optimizer = torch.optim.SGD(model.parameters(), 
        args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    train_dataset = MovieNet_SceneSeg_Dataset_Embeddings_Train(
        pkl_path=args.pkl_path_train,
        sampled_shot_num=args.seq_len,
        shuffle_p=args.sample_shulle_rate
    )
    val_dataset = MovieNet_SceneSeg_Dataset_Embeddings_Val(
        pkl_path=args.pkl_path_val,
        sampled_shot_num=args.seq_len
    )

    test_dataset = MovieNet_SceneSeg_Dataset_Embeddings_Val(
        pkl_path=args.pkl_path_test,
        sampled_shot_num=args.seq_len
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, args.train_bs, num_workers=args.workers,
        shuffle=True, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.test_bs, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.test_bs, num_workers=args.workers,
        shuffle=False, pin_memory=True, drop_last=False)
    
    train_fun = train
    test_fun = inference

    val_max_F1 = 0
    is_best = False
    test_info = {'mAP': 0, 'F1': 0}
    for epoch in range(1, args.epochs + 1):
        train_loader.dataset._shuffle_offset()
        adjust_learning_rate(args, optimizer, epoch)
        train_fun(args, model, train_loader, optimizer, epoch, criterion)
        if epoch % args.test_interval == 0 and epoch >= args.test_milestone:
            f1, map, acc_all = test_fun(args, model, val_loader)
            to_log(args, f'val set: {map, f1, acc_all}', True)
            if val_max_F1 < f1:
                val_max_F1 = f1
                f1_t, map_t, acc_all_t = test_fun(args, model, test_loader)
                test_info['mAP'] = map_t
                test_info['F1'] = f1_t
                is_best = True
                to_log(args, f'now best F1 on val is: {val_max_F1}', True)
                to_log(args, f'test set: {map_t, f1_t, acc_all_t}', True)
            else:
                is_best = False
            save_checkpoint({
                'state_dict': model.state_dict(), 'epoch': epoch,
            }, is_best=is_best, fpath=os.path.join(args.save_dir, 'checkpoint.pth.tar'))
    
    to_log(args, f'best F1 on val: {val_max_F1}', True)
    to_log(args, f"the test set mAP: {test_info['mAP']}, F1: {test_info['F1']}", True)


def train(args, model, train_loader, optimizer, epoch, criterion, log_interval=30):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        target = target.unsqueeze(-1).cuda(non_blocking=True)
        output = model(data)
        output = output.view(-1, 2)
        target = target.view(-1)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log = 'Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, 
            int(batch_idx * len(data)), len(train_loader.dataset),
            100. * batch_idx / len(train_loader)).ljust(40) + \
            'Loss: {:.6f}'.format( loss.item())
            to_log(args, log, True)

@torch.no_grad()
def inference(args, model, loader, threshhold=0.5):
    model.eval()
    corr = 0
    total = 0
    stride = args.seq_len // 2
    result_all = {}
    for batch_idx, (data, target, imdb) in enumerate(loader):
        imdb = imdb[0]
        result_all[imdb] = None
        data = data.view(-1, args.dim).cuda(non_blocking=True)
        target = target.view(-1)
        data_len = data.size(0)
        gt_len = target.size(0)
        prob_all = []
        for w_id in range(data_len//stride):
            start_pos = w_id*stride
            _data = data[start_pos:start_pos + args.seq_len].unsqueeze(0)
            output = model(_data)
            output = output.view(-1, 2)
            prob = output[:, 1]
            prob = prob[stride//2:stride+stride//2].squeeze()
            prob_all.append(prob.cpu())
        
        # metrics
        preb_all = torch.cat(prob_all,axis=0)[:gt_len].numpy()
        pre = np.nan_to_num(preb_all) > threshhold
        gt = target.cpu().numpy().astype(int)
        pre = pre.astype(int)
        idx1 = np.where(gt == 1)[0]
        idx0 = np.where(gt == 0)[0]
        idx1_p = np.where(pre == 1)[0]
        idx0_p = np.where(pre == 0)[0]
        TP = len(np.where(gt[idx1] == pre[idx1])[0])
        FP = len(np.where(gt[idx1_p] != pre[idx1_p])[0])
        TN = len(np.where(gt[idx0] == pre[idx0])[0])
        FN = len(np.where(gt[idx0_p] != pre[idx0_p])[0])
        ap = get_ap(gt, preb_all, False)
        correct = len(np.where(gt == pre)[0])
        corr += correct
        total += gt_len
        recall = TP / (TP + FN + 1e-5)
        precision = TP / (TP + FP + 1e-5)
        f1 = 2 * recall * precision / (recall + precision + 1e-5)
        result_all[imdb] = (f1, ap, recall, precision)
    mAP_all_avg = 0
    F1_all_avg = 0
    for k, v in result_all.items():
        F1_all_avg += v[0]
        mAP_all_avg += v[1]
    F1_all_avg /= len(result_all.keys())
    mAP_all_avg /= len(result_all.keys())
    return F1_all_avg, mAP_all_avg, corr / total

    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def set_log(args):
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    
    args.log_file = './output/log_' + time_str + '.txt'
    args.save_dir = args.save_dir + 'seg_checkpoints/' + time_str + '/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

def to_log(args, content, echo=False):
    with open(args.log_file, 'a') as f:
        f.writelines(content+'\n')
    if echo:
        print(content)

def adjust_learning_rate(args, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_ap(gts_raw,preds_raw,is_list=True):
    if is_list:
        gts,preds = [],[]
        for gt_raw in gts_raw:
            gts.extend(gt_raw.tolist())
        for pred_raw in preds_raw:
            preds.extend(pred_raw.tolist())
    else: 
        gts = np.array(gts_raw)
        preds = np.array(preds_raw)
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))
    # return average_precision_score(gts, preds)

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    os.makedirs(osp.dirname(fpath),exist_ok=True)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    # data
    parser.add_argument('-train', '--pkl-path-train', default='', type=str,
                    help='the path of pickle train data')

    parser.add_argument('-test', '--pkl-path-test', default='', type=str,
                    help='the path of pickle test data')

    parser.add_argument('-val', '--pkl-path-val', default='', type=str,
                    help='the path of pickle val data')
    
    parser.add_argument('--train-bs', default=12, type=int)
    parser.add_argument('--test-bs', default=1, type=int)
    parser.add_argument('--shot-num', default=10, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--gpu-id', type=str, default='0', help='gpu id')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay', dest='weight_decay')
    
    parser.add_argument('--save-dir', default='./output/', type=str,
                        help='the path of checkpoints')
    # loss weight
    parser.add_argument('--loss-weight', default=[1, 4], nargs='+', type=float,
                    help='loss weight')
    parser.add_argument('--sample-shulle-rate', default=1.0, type=float)
    parser.add_argument('--input-drop-rate', default=0.2, type=float)
    # lr schedule
    parser.add_argument('--schedule', default=[160, 180], nargs='+',
                    help='learning rate schedule (when to drop lr by a ratio)')

    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of workers')
    parser.add_argument('--dim', default=2048, type=int)
    parser.add_argument('--seq-len', default=40, type=int)
    parser.add_argument('--test-interval', default=1, type=int)
    parser.add_argument('--test-milestone', default=100, type=int)

    args = parser.parse_args()

    # assert
    assert args.seq_len % 4 == 0

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    set_log(args)
    for arg in vars(args):
        to_log(args,arg.ljust(20)+':'+str(getattr(args, arg)), True)  
    return args

if __name__ == '__main__':
    args = get_config()
    main(args)