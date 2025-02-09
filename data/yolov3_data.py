# coding:utf-8
import cv2
import numpy as np
import torch

from data import VocFormatData
from common.utils import parse_xml, normalization_coordinate, constrain_coordinate


class Yolov3Data(VocFormatData):
    def __init__(self, parameters, prepare_func, index_file='trainval'):
        data_path = parameters.get('data_path')
        classes = parameters.get('classes')
        self.img_size = parameters.get('img_size')  # tuple (w, h)
        super(Yolov3Data, self).__init__(data_path, classes, prepare_func, img_file=index_file)

    def __getitem__(self, item):
        """

        :param item:
        :return:  img, target [x, y, w, h]
        """
        img = cv2.imread(self.images[item])
        targets = parse_xml(self.targets[item], self.classes)
        if self.prepare_func:
            img, targets = self.prepare_func(img, targets)
        else:
            img_h, img_w, c = img.shape
            ratio_w = self.img_size[0] / img_w
            ratio_h = self.img_size[1] / img_h
            targets[:, :-1:2] *= ratio_w
            targets[:, 1:-1:2] *= ratio_h
            img = cv2.resize(img, self.img_size)
        targets = normalization_coordinate(targets, img_shape=img.shape)
        target = constrain_coordinate(0, 1, targets)
        return img, target

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        new_targets = []
        for i, boxes in enumerate(targets):
            value = np.empty(boxes.shape[0])
            value.fill(i)
            new_targets.append(np.insert(boxes, 0, values=value, axis=1))
        new_targets = torch.tensor(np.vstack(new_targets), dtype=torch.float32)
        images = torch.tensor(imgs, dtype=torch.float32)
        images = images.permute(0, 3, 1, 2).contiguous()
        return images, new_targets


if __name__ == '__main__':
    data_path = '/home/lintaowx/data/VOCdevkit/VOC2007'
    classes = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
    parameters = {
        'data_path': data_path,
        'classes': classes,
        'img_size':  (640, 640)
    }
    data_set = Yolov3Data(parameters, None)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(data_set, 32, shuffle=False, num_workers=1, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, a in enumerate(dataloader):
            print(a)
