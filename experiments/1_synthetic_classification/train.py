import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary

from shapenet_illuminants import ShapeNetIlluminants
import resnet

def train(epoch, model, criterion, optimizer, dataloader):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    return 100. * correct / len(dataloader.dataset)

def test(model, dataloader):
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in dataloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(dataloader.dataset)
    return acc.item()

def main(args):

    args.cuda = torch.cuda.is_available()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    # Load model
    if 'resnet' in args.model:
        model = getattr(resnet, args.model)(num_classes=10, pretrained=args.pretrained, invariant=args.invariant, scale=args.scale)
    else:
        raise AssertionError('Only ResNets implemented.')
    if args.cuda:
        model.cuda()
    summary(model, (3,224,224))

    # Transformations
    p_hflip = 0.5 if args.hflip else 0
    tr_train = transforms.Compose([transforms.RandomCrop(224, padding=args.rc),
                                   transforms.RandomHorizontalFlip(p=p_hflip),
                                   transforms.ColorJitter(brightness=args.jitter,
                                                          contrast=args.jitter,
                                                          saturation=args.jitter,
                                                          hue=args.jitter),
                                   transforms.RandomRotation(args.rr, resample=3),
                                   transforms.ToTensor()])
    tr_test = transforms.Compose([transforms.ToTensor()])

    # Add normalization for RGB input
    if not args.invariant:
        tr_train = transforms.Compose([tr_train,transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        tr_test = transforms.Compose([tr_test,transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    # Load ShapeNet-Illuminants dataset
    x_train = ShapeNetIlluminants(args.root, split='train', transform=tr_train)
    x_val = ShapeNetIlluminants(args.root, split='val', transform=tr_test)
    x_test = []
    x_test.append(ShapeNetIlluminants(args.root, split='test_2500K', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_4000K', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_6500K', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_12000K', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_20000K', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_darkest', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_dark', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_light', transform=tr_test))
    x_test.append(ShapeNetIlluminants(args.root, split='test_lightest', transform=tr_test))

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = []
    for split in x_test:
        test_loader.append(torch.utils.data.DataLoader(split, batch_size=args.test_batch_size, shuffle=False, **kwargs))

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)
    if args.pretrained:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,70,90], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150], gamma=0.1)

    s = 0.

    # Training loop
    for epoch in range(1, args.epochs + 1):

        train_acc = train(epoch, model, criterion, optimizer, train_loader)
        if args.invariant:
            s = model.ciconv.scale.item()
        val_acc = test(model, val_loader)
        print('Epoch: [{:d}/{:d}] \t Train acc: {:.2f}% \t Val acc: {:.2f}% \t'
              'LR: {:.1e} \t Scale: {:.2f}'.format(epoch, args.epochs, train_acc, val_acc,
                                                   optimizer.param_groups[0]['lr'], s))

        scheduler.step()

    # Test with last weights
    if args.invariant:
        s = model.ciconv.scale.item()
    r_last = []
    for loader in test_loader:
        r_last.append(test(model, loader))
    print('== Results summary - scale={:.2f} =='.format(s))
    print('2500K, 4000K, 6500K, 12000K, 20000K, darkest, dark, normal, light, lightest')
    print(r_last[:7]+[r_last[2]]+r_last[7:]) # 6500K and normal are the same


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='CODaN training and evaluation')
    parser.add_argument('--root', type=str, default='../data/shapenet-illuminants',
                        help='path to synthetic dataset')
    parser.add_argument('--test', type=str, default=None, help='model weights to evaluate')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model (any resnet)')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--scale', metavar='0.0', default=0.,
                        help='initial value for scale', type=float)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=175, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--rc', type=int, default=0, metavar='RC',
                        help='random cropping (default: 0')
    parser.add_argument('--rr', type=int, default=20, metavar='RR',
                        help='random rotations (default: 20')
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--jitter', type=float, default=0, metavar='J',
                        help='color jitter augmentation (default: 0.0)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize from imagenet pretraining')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    args = parser.parse_args()
    print(args)

    main(args)
