# coding:utf-8
import os

import numpy as np
from tqdm import tqdm

from data import DataBases


class VocFormatData(DataBases):

    def __init__(self, data_path, classes, prepare_func, img_file='trainval'):
        super(VocFormatData, self).__init__(data_path)
        self.classes = classes
        self.classes_num = len(classes)
        self.prepare_func = prepare_func
        self._anno_path = os.path.join(self.data_path, 'Annotations')
        self._img_path = os.path.join(self.data_path, 'JPEGImages')
        self._file_index = os.path.join(self.data_path, 'ImageSets', 'Main', img_file + '.txt')
        with open(self._file_index, 'r') as f:
            self.img_files = [(os.path.join(self._img_path, line.strip() + '.jpg'),
                               os.path.join(self._anno_path, line.strip() + '.xml'))
                              for line in tqdm(f.readlines(), desc='Reading labels') if line.strip() != ""]
            # self.img_files = [(os.path.join(self._img_path, line.strip() + '.jpg'),
            #                    parse_xml(os.path.join(self._anno_path, line.strip() + '.xml'), self.classes))
            #                   for line in tqdm(f.readlines(), desc='Reading labels')]

            self.img_files = np.array(self.img_files)
            np.random.shuffle(self.img_files)
        self.img_files = self.img_files.T
        self.images = self.img_files[0]
        self.targets = self.img_files[1]

    def __len__(self):
        return len(self.images)

    def collate_fn(self, *any):
        pass


if __name__ == '__main__':
    # data_path = '~/data/object/voc/VOCdevkit/VOC2012'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'
    classes = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
    data_set = VocFormatData(data_path, classes, None, img_file='trainval')
    from torch.utils.data import DataLoader

    dataloader = DataLoader(data_set, 32, shuffle=False, num_workers=1, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, a in enumerate(dataloader):
            print(a)

    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # np.random.shuffle(a)
    # print(123)
