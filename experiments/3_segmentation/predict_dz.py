# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:37 2019 by Attila Lengyel - attila@lengyel.nl
"""

from utils.helpers import get_test_trans, vislbl
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet

import numpy as np
import os, cv2, torch

def main(args):

    if not os.path.isdir('pred_dz'):
        os.makedirs('pred_dz/col')
        os.makedirs('pred_dz/submit/labelTrainIds')
        os.makedirs('pred_dz/submit/labelTrainIds_invalid')
        os.makedirs('pred_dz/submit/confidence')

    # Define data transformation
    mean = (0.485, 0.456, 0.406) if not args.invariant else None
    std = (0.229, 0.224, 0.225) if not args.invariant else None
    target_size = (512,1024)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    dz_path = '/dark_zurich/path/here'
    testset_dz = DarkZurichDataset(dz_path, split='test', transforms=test_trans)
    dataloader = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(DarkZurichDataset.validClasses)
    model = RefineNet(num_classes, pretrained=False, invariant=args.invariant)

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Load model weights
    model.load_state_dict(torch.load(args.weights)['model_state_dict'])

    # Disable tracking BN stats for AdaBN-style evaluation
    if args.adabn:
        print('Evaluating with batch statistics...')
        for module in model.modules():
            if 'BatchNorm2d' in module.__class__.__name__:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

    print(model)

    # set model in training mode
    model.eval()

    with torch.no_grad():
        for epoch_step, data in enumerate(dataloader):
            if len(data) == 2:
                inputs, filepath = data
            else:
                inputs, _, filepath = data

            # input resolution
            inputs = inputs.float().cuda()

            # forward
            outputs = model(inputs).cpu()
            preds = torch.argmax(outputs, 1)
            confs = torch.max(torch.nn.functional.softmax(outputs,dim=1), 1)[0]

            for i in range(inputs.size(0)):
                filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                # Save predictions
                pred = vislbl(preds[i,:,:], DarkZurichDataset.mask_colors)
                cv2.imwrite('pred_dz/col/{}.png'.format(filename),pred[:,:,::-1])

                pred = cv2.resize(preds[i,:,:].numpy().astype('uint8'),(1920,1080),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('pred_dz/submit/labelTrainIds/{}_id.png'.format(filename),pred)
                cv2.imwrite('pred_dz/submit/labelTrainIds_invalid/{}_id.png'.format(filename),pred)
                # Dummy confidence map
                conf = cv2.resize((65535*confs[i,:,:].numpy()).astype(np.uint16),(1920,1080),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('pred_dz/submit/confidence/{}_id.png'.format(filename),conf)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('weights', type=str, default=None,
                        help='resume training from checkpoint')
    parser.add_argument('--invariant', type=str, default=None,
                        help='invariant (E,W,C,N,H)')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--adabn', action='store_true', default=False,
                        help='use batch statistics')
    args = parser.parse_args()

    main(args)
