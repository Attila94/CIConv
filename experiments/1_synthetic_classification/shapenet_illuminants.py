from PIL import Image
import os
import os.path
import numpy as np

from torchvision.datasets.vision import VisionDataset

class ShapeNetIlluminants(VisionDataset):
    """`CODaN <https://github.com/Attila94/CODaN>`_ Dataset.

    Args:
        data (string, optional): Location of the downloaded .tar.bz2 files.
        split (string, optional): Define which dataset split to use. Must be one of
            'train', 'val', 'test_day', 'test_night'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
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

        # Unpack archives
        if not os.path.isdir(os.path.join(root,split)):
            raise AssertionError('Dataset missing.')
            # # Join .tar.bz2 parts files for training split
            # if split == 'train' and not os.path.exists(os.path.join(root,'codan_train.tar.bz2')):
            #     with open(os.path.join(root,'codan_train.tar.bz2'), 'wb') as f_out:
            #         for i in range(3):
            #             fpath = os.path.join(root,'codan_train.tar.bz2.part{}'.format(i))
            #             with open(fpath, 'rb') as f_in:
            #                 f_out.write(f_in.read())
            #             os.remove(fpath)
            # # Unpack tar
            # tarpath = os.path.join(root,'codan_'+split+'.tar.bz2')
            # with tarfile.open(tarpath) as tar:
            #     print('Unpacking {} split.'.format(split))
            #     tar.extractall(path='./data')
        # else:
        #     print('Dataset {} split already extracted.'.format(split))

        # loop through split directory, load all images in memory using PIL
        if split in ['train','val']:
            for i, c in enumerate(cls_dict.keys()):
                imsar = np.load(os.path.join(split,c+'.npz'))['arr_0']
                for j in range(len(imsar)):
                    self.data.append(Image.fromarray(imsar[j]))
                    self.targets.append(i)
        else:
            for i, c in enumerate(cls_dict.keys()):
                im_dir = os.path.join(root,split,c)
                ims = os.listdir(im_dir)
                ims = [im for im in ims if '.png' in im]  # remove any system files
                assert len(ims) == nfiles, 'Found {} files instead of {}'.format(len(ims),nfiles)

                for im in ims:
                    # img = Image.open(os.path.join(im_dir,im))
                    # self.data.append(img.copy())
                    # img.close()
                    self.data.append(os.path.join(im_dir,im))
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
        # Open image
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
