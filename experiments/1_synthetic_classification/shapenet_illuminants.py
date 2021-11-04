from PIL import Image
import os
import os.path
import numpy as np

from torchvision.datasets.vision import VisionDataset

class ShapeNetIlluminants(VisionDataset):
    """`ShapeNet Illuminants <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        root (string, optional): Location of the downloaded dataset.
        split (string, optional): Define which dataset split to use. Must be one of
            'train', 'val', 'test', 'test_2500K', 'test_4000K', 'test_6500K', 'test_12000K',
            'test_20000K', 'test_dark', 'test_darkest', 'test_light', 'test_lightest'.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root='./data', split='train', transform=None, target_transform=None):
        super(ShapeNetIlluminants, self).__init__(root, transform, target_transform)

        cls_dict = {'02691156': 'airplane',
                    '02933112': 'cabinet',
                    '03001627': 'chair',
                    '03691459': 'loudspeaker',
                    '04379243': 'table',
                    '02828884': 'bench',
                    '02958343': 'car',
                    '03636649': 'lamp',
                    '04256520': 'sofa',
                    '04530566': 'watercraft'}

        split_list = ['train','val','test','test_2500K','test_4000K','test_6500K','test_12000K',
                      'test_20000K','test_dark','test_darkest','test_light','test_lightest']
        assert split in split_list, 'Invalid split.'
        if split == 'test': split = 'test_6500K'  # 6500K is same as train and val splits
        split = split.replace('_', '/')
        if split == 'train':
            nfiles = 1000
        elif split == 'val':
            nfiles = 100
        else:
            nfiles = 300

        self.split = split  # dataset split
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        # Check if dataset exists
        if not os.path.isdir(root):
            raise AssertionError('Dataset missing.')

        for i, c in enumerate(cls_dict.keys()):
            # load train and val split from npy
            if split in ['train','val']:
                # create .npy files for train and val split for faster loading
                if not os.path.exists(os.path.join(root,split,c+'.npy')):
                    print('Generating .npy file for split {}, class {}.'.format(split, c))
                    # get list of files in class dir
                    ims = os.listdir(os.path.join(root,split,c))
                    ims = [im for im in ims if '.png' in im]  # remove any system files
                    assert len(ims) == nfiles, 'Found {} files instead of {}'.format(len(ims),nfiles)
                    # load files into memory                    
                    im_array = []
                    for im in ims:
                        img = Image.open(os.path.join(root,split,c,im)).convert('RGB')
                        im_array.append(np.array(img.copy()))
                        img.close()
                    # save array as .npy
                    im_array = np.asarray(im_array)
                    np.save(os.path.join(root,split,c), im_array)

                # load data from .npy files into memory
                imsar = np.load(os.path.join(root,split,c+'.npy'))
                for j in range(len(imsar)):
                    self.data.append(Image.fromarray(imsar[j]))
                    self.targets.append(i)

            # load test splits on the fly
            else:
                # get list of files in class dir
                ims = os.listdir(os.path.join(root,split,c))
                ims = [im for im in ims if '.png' in im]  # remove any system files
                assert len(ims) == nfiles, 'Found {} files instead of {}'.format(len(ims),nfiles)
                for im in ims:
                    self.data.append(os.path.join(os.path.join(root,split,c),im))
                    self.targets.append(i)

        print('Dataset {} split loaded.'.format(split))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # Load images on the go for test splits
        if self.split not in ['train','val']:
            img_tmp = Image.open(img).convert('RGB')
            img = img_tmp.copy()
            img_tmp.close()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
