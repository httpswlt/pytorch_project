# coding:utf-8
import os

import cv2
import numpy as np
import torch

from data import DataBases, ImageFactory


class ClassifyData(DataBases):
    """
        the dir structure should is:
        -- train
            -- 01
                -- image1.jpg
                -- image2.jpg
                ...
            -- 02
                -- image1.jpg
                -- image2.jpg
                ...
            ...
    """
    
    def __init__(self, data_path, prepare_func):
        super(ClassifyData, self).__init__(data_path)
        self.classes = os.listdir(data_path)
        self.prepare_func = prepare_func
        assert os.path.exists(data_path)
        assert self.prepare_func is not None
        
        self.img_files = []
        for i, class_type in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_type)
            for image in os.listdir(class_path):
                image = image.strip().split('\n')[0]
                self.img_files.append([os.path.join(class_path, image), i])
        
        self.img_files = np.array(self.img_files)
        np.random.shuffle(self.img_files)
        self.img_files = self.img_files.T
        self.images = self.img_files[0]
        self.targets = self.img_files[1].astype(np.int32)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        """

        :param item:
        :return:  img, target [x, y, w, h]
        """
        img = cv2.imread(self.images[item])
        targets = self.targets[item]
        img, targets = self.prepare_func(img, targets)
        return img, targets
    
    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        new_targets = torch.tensor(targets, dtype=torch.long)
        images = torch.tensor(imgs, dtype=torch.float32)
        images = images.permute(0, 3, 1, 2).contiguous()
        return images, new_targets


class PrepareData(ImageFactory):
    def __init__(self):
        super(PrepareData, self).__init__()
    
    def run(self, img, targets):
        """

        :param img:
        :param targets:
        :return:
        """
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, self.img_size)
        return img, targets


if __name__ == '__main__':
    # data_path = '~/data/object/voc/VOCdevkit/VOC2012'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'
    prepare_data = PrepareData()
    prepare_data.set_image_size((600, 200))
    data_set = ClassifyData(data_path, prepare_data)
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(data_set, 32, shuffle=True, num_workers=6, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, a in enumerate(dataloader):
            print(a)
    
    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # np.random.shuffle(a)
    # print(123)
