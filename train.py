import os
import sys
import argparse
import time
import random
import logging
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader


from dataloader.dataset import ColorDatasetTrain, ColorDatasetVal
from models.model import Color_model
from models.layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer

from utils.utils import AverageMeter, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

def main():
    parser = argparse.ArgumentParser(description='Colorization!')
    ## Optimizer
    parser.add_argument('--gpu', default='1', help='gpu id')
    parser.add_argument('--num_epoch', default=15, type=int, help='training epoch')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=40, type=int, help='batch size')
    ## Dataset
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--crop_size', default = 224, type = int, help = 'size for randomly cropping images')
    parser.add_argument('--data_root', type=str, default='/home/ubuntu/lsz/dataset/imagenet/ILSVRC/Resize',
                        help='path to dataset splits data folder')
    parser.add_argument('--dataset', default='imagenet', type=str,)
    ## Checkpoint
    parser.add_argument('--save_step', type = int, default = 1000, help = 'step size for saving trained models')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    ## Utils
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='eval')

    global args
    args = parser.parse_args()
    
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    ## save logs
    if args.savename=='default':
        args.savename = 'color_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))
    
    # Dataset
    train_transform = transforms.Compose([
                         # transforms.Scale(args.size),
                         transforms.RandomCrop(args.crop_size),
                         transforms.RandomHorizontalFlip(),
                        ])
    # val_transform = transforms.Compose([
    #                      transforms.Scale(args.size),
    #                     ])
    val_transform = None
    train_dataset = ColorDatasetTrain(data_root=args.data_root, split='train', transform=train_transform)
    val_dataset = ColorDatasetVal(data_root=args.data_root, split='val', transform=val_transform)
    test_dataset = ColorDatasetVal(data_root=args.data_root, split='test', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=False, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=4)
    
    ## Model
    model = nn.DataParallel(Color_model()).cuda()
    encode_layer = NNEncLayer()
    boost_layer = PriorBoostLayer()
    nongray_mask = NonGrayMaskLayer()
    
    if args.pretrain:
        model=load_pretrain(model,args,logging)
    if args.resume:
        model=load_resume(model,args,logging)

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))
    visu_param = list(model.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    print('model parameters:', sum_visu)
    
    ## Loss and Optimizer
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    optimizer = torch.optim.Adam(
            [{'params': visu_param, 'lr': args.lr/10.},], 
            lr=args.lr, 
            betas=(0.9, 0.99),
            weight_decay=0.001)
    
    ## training and testing
    best_accu = -float('Inf')
    if args.test:
        test_epoch(test_loader, model)
    elif args.eval:
        validate_epoch(val_loader, model)
    else:
        step = 0
        for epoch in range(args.nb_epoch):
            #--------------------------------------------------------
            batch_time = AverageMeter()
            losses = AverageMeter()
            model.train()
            end = time.time()
            for batch_idx, (images, img_ab) in enumerate(train_loader):
                adjust_learning_rate(optimizer, step)
                images = images.unsqueeze(1).float().cuda()
                img_ab = img_ab.float()  # [bs, 2, 56, 56]
                
                ## Preprocess data
                encode, max_encode = encode_layer.forward(img_ab)  # Paper Eq(2) Z空间ground-truth的计算
                targets = torch.Tensor(max_encode).long().cuda()
                boost = torch.Tensor(boost_layer.forward(encode)).float().cuda()  # Paper Eq(3)-(4), [bs, 1, 56, 56], 每个空间位置的ab概率
                mask = torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()  # ab通道数值和小于5的空间位置不计算loss, [bs, 1, 1, 1]
                boost_nongray = boost * mask
                
                outputs = model(images)
                
                # compute loss
                loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), images.size(0))
            
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                step += 1

                if step % args.print_freq == 0:
                    print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'vis_lr {vis_lr:.8f}\t' \
                        .format( \
                            epoch, batch_idx, len(train_loader), \
                            loss=losses, vis_lr = optimizer.param_groups[0]['lr'])
                    print(print_str)
                    logging.info(print_str)
                    
                if step % args.save_step == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': losses.avg,
                        'optimizer' : optimizer.state_dict(),
                        }, False, args, filename='colornet_'+str(step))
            #--------------------------------------------------------
            

def validate_epoch(val_loader, model, mode='val'):
    pass
        

def test_epoch(val_loader, model, mode='test'):
    pass


if __name__ == "__main__":
    main()
