# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:40:51 2019 by Attila Lengyel - attila@lengyel.nl
"""

from PIL import Image
import numpy as np

from torchvision.datasets import Cityscapes
from torch.utils.data import Subset

### Cityscapes
class CityscapesExt(Cityscapes):
    
    voidClass = 19
    
    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass
    
    # Convert train_ids to colors
    maskColors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    maskColors.append([0,0,0])
    maskColors = np.array(maskColors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return image, target, filepath
    
    def valtest_split(self, p_split, random_seed):
        assert self.split == 'val', 'Dataset split must be "val".'
        # Split test into val/test according to VALIDATION_SPLIT
        num_test = len(self)
        indices = list(range(num_test))
        split = int(np.floor(p_split * num_test))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        valset = Subset(self, indices[:split])
        testset = Subset(self, indices[split:])
        return valset, testset