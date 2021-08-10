# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:37 2019 by Attila Lengyel - attila@lengyel.nl
"""

from utils.helpers import gen_train_dirs, plot_confusion_matrix, get_train_trans, get_test_trans
from utils.routines import train_epoch, evaluate
from datasets.cityscapes_ext import CityscapesExt
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

import shutil, time, random
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    # Configure dataset paths here
    cs_path = 'datasets/Cityscapes/'
    nd_path = 'datasets/NighttimeDrivingTest/'
    dz_path = 'datasets/Dark_Zurich_val_anon/'

    print('--- Training args ---')
    print(args)

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Generate training directories
    gen_train_dirs()

    # Generate log files
    with open('logs/log_batch.csv', 'a') as batch_log:
        batch_log.write('epoch, epoch step, train loss, avg train loss, train acc, avg train acc\n')
    with open('logs/log_epoch.csv', 'a') as epoch_log:
        epoch_log.write('epoch, train loss, val loss day, train acc, val acc day, test night acc, miou, miou nd, miou dz, learning rate, scale \n')

    # Initialize metrics
    best_miou = 0.0
    metrics = {'train_loss': [],
               'train_acc': [],
               'val_acc_cs': [],
               'val_loss_cs': [],
               'miou_cs': [],
               'test_acc_nd': [],
               'test_loss_nd': [],
               'miou_nd': [],
               'test_acc_dz': [],
               'test_loss_dz': [],
               'miou_dz': []}
    start_epoch = 0

    # Define data transformation
    mean = (0.485, 0.456, 0.406) if not args.invariant else None
    std = (0.229, 0.224, 0.225) if not args.invariant else None
    target_size = (512,1024)
    crop_size = (384,768) if args.rc else None
    train_trans = get_train_trans(mean, std, target_size, crop_size, args.jitter, args.scale, args.hflip)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    trainset = CityscapesExt(cs_path, split='train', target_type='semantic', transforms=train_trans)
    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
    testset_dz = DarkZurichDataset(dz_path, transforms=test_trans)

    # Use mini-dataset for debugging purposes
    if args.xs:
        trainset = Subset(trainset, list(range(5)))
        valset = Subset(valset, list(range(5)))
        testset_nd = Subset(testset_nd, list(range(5)))
        testset_dz = Subset(testset_dz, list(range(5)))
        print('WARNING: XS_DATASET SET TRUE')

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
    model = RefineNet(num_classes, pretrained=args.pretrained, invariant=args.invariant)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)

    # Resume training from checkpoint
    if args.resume:
        print('Resuming training from {}.'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        metrics = checkpoint['metrics']
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch']+1

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    since = time.time()

    for epoch in range(start_epoch, args.epochs):

        # Train
        print('--- Training ---')
        train_loss, train_acc = train_epoch(dataloaders['train'], model, criterion, optimizer, epoch, void=CityscapesExt.voidClass)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)

        # Validate
        print('--- Validation - daytime ---')
        val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(dataloaders['val'],
            model, criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        print('--- Validation - Nighttime Driving ---')
        test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = evaluate(dataloaders['test_nd'],
            model, criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        print('--- Validation - Dark Zurich ---')
        test_acc_dz, test_loss_dz, miou_dz, confmat_dz, iousum_dz = evaluate(dataloaders['test_dz'],
            model, criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        metrics['val_acc_cs'].append(val_acc_cs)
        metrics['val_loss_cs'].append(val_loss_cs)
        metrics['miou_cs'].append(miou_cs)
        metrics['test_acc_nd'].append(test_acc_nd)
        metrics['test_loss_nd'].append(test_loss_nd)
        metrics['miou_nd'].append(miou_nd)
        metrics['test_acc_dz'].append(test_acc_dz)
        metrics['test_loss_dz'].append(test_loss_dz)
        metrics['miou_dz'].append(miou_dz)

        # Write logs
        scale = model.module.resnet.ciconv.scale.item() if args.invariant else None
        with open('logs/log_epoch.csv', 'a') as epoch_log:
            epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                    epoch, train_loss, val_loss_cs, train_acc, val_acc_cs, test_acc_nd, miou_cs, miou_nd, miou_dz, lr, scale))
        with open('logs/class_iou.txt', 'w') as ioufile:
            ioufile.write(iousum_cs)
        # Plot confusion matrices
        cm_title = 'mIoU : {:.3f}, acc : {:.3f}'.format(miou_cs, val_acc_cs)
        plot_confusion_matrix(confmat_cs,CityscapesExt.classLabels,normalize=True,title=cm_title).savefig('logs/confmat_cs.pdf', bbox_inches='tight')
        cm_title = 'mIoU : {:.3f}, acc : {:.3f}'.format(miou_nd, test_acc_nd)
        plot_confusion_matrix(confmat_nd,CityscapesExt.classLabels,normalize=True,title=cm_title).savefig('logs/confmat_nd.pdf', bbox_inches='tight')
        cm_title = 'mIoU : {:.3f}, acc : {:.3f}'.format(miou_dz, test_acc_dz)
        plot_confusion_matrix(confmat_dz,CityscapesExt.classLabels,normalize=True,title=cm_title).savefig('logs/confmat_dz.pdf', bbox_inches='tight')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'metrics': metrics,
            }, 'weights/checkpoint.pth.tar')

        # Save best model to file
        if miou_cs > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou_cs))
            best_miou = miou_cs
            best_acc = val_acc_cs # acc corresponding to the best miou
            shutil.copy('logs/confmat_cs.pdf', 'logs/best_confmat_cs.pdf') # save confmat
            shutil.copy('logs/confmat_nd.pdf', 'logs/best_confmat_nd.pdf') # save confmat
            shutil.copy('logs/confmat_dz.pdf', 'logs/best_confmat_dz.pdf') # save confmat
            shutil.copy('logs/class_iou.txt', 'logs/best_class_iou.txt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, 'weights/best_weights.pth.tar')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best mIoU: {:4f}, corresponding val acc: {:4f}'.format(best_miou,best_acc))

    # Plot learning curves
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('miou')
    ln1 = ax1.plot(x, metrics['miou_cs'], color='tab:red')
    ln2 = ax1.plot(x, metrics['miou_nd'], color='tab:green')
    ln3 = ax1.plot(x, metrics['miou_dz'], color='tab:blue')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln4 = ax2.plot(x, metrics['val_acc_cs'], color='tab:red', linestyle='dashed')
    ln5 = ax2.plot(x, metrics['test_acc_nd'], color='tab:green', linestyle='dashed')
    ln6 = ax2.plot(x, metrics['test_acc_dz'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4+ln5+ln6
    plt.legend(lns, ['CS mIoU','ND mIoU','DZ mIoU','CS Accuracy','ND Accuracy', 'DZ Accuracy'])
    plt.tight_layout()
    plt.savefig('logs/learning_curve.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--init-scale', metavar='1.0', default=[1.0], type=float,
                        help='initial value for scale')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training from checkpoint')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--rc', action='store_true', default=False,
                        help='perform random cropping')
    parser.add_argument('--jitter', type=float, default=0.0, metavar='J',
                        help='color jitter augmentation (default: 0.0)')
    parser.add_argument('--scale', type=float, default=0.0, metavar='J',
                        help='random scale augmentation (default: 0.0)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize feature extractor from imagenet pretrained weights')
    parser.add_argument('--xs', action='store_true', default=False,
                        help='use small dataset subset for debugging')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    args = parser.parse_args()

    main(args)
