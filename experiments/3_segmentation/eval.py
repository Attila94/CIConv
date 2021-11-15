# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:37 2019 by Attila Lengyel - attila@lengyel.nl
"""

from utils.helpers import get_test_trans
from utils.routines import evaluate
from datasets.cityscapes_ext import CityscapesExt
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet

import torch
import torch.nn as nn

def main(args):

    print(args)

    # Define data transformation
    mean = (0.485, 0.456, 0.406) if not args.invariant else None
    std = (0.229, 0.224, 0.225) if not args.invariant else None
    target_size = (512,1024)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    cs_path = '/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/Cityscapes/'
    nd_path = './NighttimeDrivingTest/'
    dz_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/attilalengyel/datasets/Dark_Zurich_val_anon/'
#    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
#    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
#    testset_dz = DarkZurichDataset(dz_path, transforms=test_trans)

    dataloaders = {}
#    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
#    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
#    dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
    model = RefineNet(num_classes, pretrained=False, invariant=args.invariant)

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Load weights from checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # Disable tracking BN stats for AdaBN-style evaluation
    if args.adabn:
        print('Evaluating with batch statistics...')
        for module in model.modules():
            if 'BatchNorm2d' in module.__class__.__name__:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    # print(model)

    # Validate
 #   print('--- Validation - daytime ---')
 #   val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(dataloaders['val'],
 #       model, criterion, 0, CityscapesExt.classLabels, CityscapesExt.validClasses,
 #       void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
    print('--- Validation - Nighttime Driving ---')
    test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = evaluate(dataloaders['test_nd'],
        model, criterion, 0, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
 #   print('--- Validation - Dark Zurich ---')
 #   test_acc_dz, test_loss_dz, miou_dz, confmat_dz, iousum_dz = evaluate(dataloaders['test_dz'],
 #       model, criterion, 0, CityscapesExt.classLabels, CityscapesExt.validClasses,
 #       void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('weight', type=str, default=None,
                        help='load weight file')
    parser.add_argument('--model', type=str, default='refinenet',
                        help='model (refinenet or deeplabv3)')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--adabn', action='store_true', default=False,
                        help='use batch statistics')
    args = parser.parse_args()

    main(args)
