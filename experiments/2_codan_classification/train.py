import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary

from codan import CODaN
import resnet
import preproc

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

def test(model, criterion, dataloader, track_running_stats=True):
    model.eval()

    # Disable tracking BN stats for AdaBN-style evaluation
    if track_running_stats is False:
        for module in model.modules():
            if 'BatchNorm2d' in module.__class__.__name__:
                module.track_running_stats = track_running_stats
                module.running_mean = None
                module.running_var = None

    test_loss = 0
    correct = 0
    for data, target in dataloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataloader.dataset)
    acc = 100. * correct / len(dataloader.dataset)
    return acc

def main(args):

    args.cuda = torch.cuda.is_available()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    tstr = str(int(time.time()))
    print('Weights UID: {}'.format(tstr))

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    # Load model
    if 'resnet' in args.model:
        model = getattr(resnet, args.model)(num_classes=10, pretrained=args.pretrained, invariant=args.invariant, scale=args.scale)
    else:
        raise AssertionError('Only ResNets implemented.')

    # Load weights
    if args.weights:
        w = torch.load(args.weights)
        del w['fc.weight']
        del w['fc.bias']
        r = model.load_state_dict(w, strict=False)
        print(r)
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
                                   transforms.ToTensor(),
                                   preproc.norm()])
    tr_test = transforms.Compose([transforms.ToTensor(),
                                  preproc.norm()])
    # Add custom preprocessing function
    if args.preproc:
        if args.preproc == 'luminance':
            tr_train = transforms.Compose([tr_train, transforms.Grayscale(num_output_channels=3)])
            tr_test = transforms.Compose([tr_test, transforms.Grayscale(num_output_channels=3)])
        else:
            tr_train = transforms.Compose([tr_train, getattr(preproc, args.preproc)(),
                                           preproc.norm()])
            tr_test = transforms.Compose([tr_test, getattr(preproc, args.preproc)(),
                                          preproc.norm()])
    # Add normalization for RGB input
    if not args.invariant and not args.preproc:
        tr_train = transforms.Compose([tr_train,transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        tr_test = transforms.Compose([tr_test,transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    # Load CODaN dataset
    if not args.test:
        x_train = CODaN(args.root, split='train', transform=tr_train)
        x_val = CODaN(args.root, split='val', transform=tr_test)
        train_loader = torch.utils.data.DataLoader(x_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(x_val, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    x_test_day = CODaN(args.root, split='test_day', transform=tr_test)
    x_test_night = CODaN(args.root, split='test_night', transform=tr_test)
    test_day_loader = torch.utils.data.DataLoader(x_test_day, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_night_loader = torch.utils.data.DataLoader(x_test_night, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)
    if args.pretrained:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,70,90], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150], gamma=0.1)

    s = 0.

    # Test model
    if args.test:
        model.load_state_dict(torch.load(args.test))
        if args.invariant:
            s = model.ciconv.scale.item()
        acc_d = test(model, criterion, test_day_loader)
        acc_n = test(model, criterion, test_night_loader)
        print('[Test] \t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))
        acc_d = test(model, criterion, test_day_loader, track_running_stats=False)
        acc_n = test(model, criterion, test_night_loader, track_running_stats=False)
        print('[Test - batch stats] \t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))
        return

    # Training loop
    best_acc = 0
    for epoch in range(1, args.epochs + 1):

        train_acc = train(epoch, model, criterion, optimizer, train_loader)
        if args.invariant:
            s = model.ciconv.scale.item()
        val_acc = test(model, criterion, val_loader)
        print('Epoch: [{:d}/{:d}] \t Train acc: {:.2f}% \t Val acc: {:.2f}% \t'
              'LR: {:.1e} \t Scale: {:.2f}'.format(epoch, args.epochs, train_acc, val_acc,
                                                   optimizer.param_groups[0]['lr'], s))

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            # Save model
            outpth = './codan_{}.pth.tar'.format(tstr)
            torch.save(model.state_dict(),outpth)

    # Test with last weights
    if args.invariant:
        s = model.ciconv.scale.item()
    acc_d = test(model, criterion, test_day_loader)
    acc_n = test(model, criterion, test_night_loader)
    print('[Final] \t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))
    # acc_d = test(model, criterion, test_day_loader, track_running_stats=False)
    # acc_n = test(model, criterion, test_night_loader, track_running_stats=False)
    # print('[Final] \t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))

    # Test with best weights
    model.load_state_dict(torch.load(outpth))
    if args.invariant:
        s = model.ciconv.scale.item()
    acc_d = test(model, criterion, test_day_loader)
    acc_n = test(model, criterion, test_night_loader)
    print('[Best] \t\t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))
    # acc_d = test(model, criterion, test_day_loader, track_running_stats=False)
    # acc_n = test(model, criterion, test_night_loader, track_running_stats=False)
    # print('[Best] \t\t Day: {:.3f} \t Night: {:.3f} \t Scale: {:.2f}'.format(acc_d, acc_n, s))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='CODaN training and evaluation')
    parser.add_argument('--root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--weights', type=str, default=None, help='load weights for training')
    parser.add_argument('--test', type=str, default=None, help='model weights to evaluate')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model (any resnet)')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--preproc', type=str, default=None,
                        help='custom preprocessing function')
    parser.add_argument('--scale', metavar='0.0', default=0.,
                        help='initial value for scale', type=float)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=175, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--rc', type=int, default=0, metavar='RC',
                        help='random cropping (default: 0')
    parser.add_argument('--rr', type=int, default=0, metavar='RR',
                        help='random rotations (default: 0')
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--jitter', type=float, default=0.0, metavar='J',
                        help='color jitter augmentation (default: 0.0)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize from imagenet pretraining')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    args = parser.parse_args()
    print(args)

    main(args)
