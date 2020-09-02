# coding:utf-8
import os

import numpy as np
from tqdm import tqdm
import cv2

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

    def __init__(self, data_path, prepare_func=None):
        super(ClassifyData, self).__init__(data_path)
        self.classes = os.listdir(data_path)
        self.prepare_func = prepare_func
        assert os.path.exists(data_path)
        assert self.prepare_func is not None

        for i, class_type in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_type)
            for image in os.listdir(class_path):
                self.img_files = [(os.path.join(class_path, image.strip()), i)]
                self.img_files = np.array(self.img_files)
                np.random.shuffle(self.img_files)

        self.img_files = self.img_files.T
        self.images = self.img_files[0]
        self.targets = self.img_files[1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """

        :param item:
        :return:  img, target [x, y, w, h]
        """
        img = cv2.imread(self.images[item])
        # if self.prepare_func:
        #     img, targets = self.prepare_func(img, targets)
        # else:
        #     img_h, img_w, c = img.shape
        #     ratio_w = self.img_size[0] / img_w
        #     ratio_h = self.img_size[1] / img_h
        #     targets[:, :-1:2] *= ratio_w
        #     targets[:, 1:-1:2] *= ratio_h
        #     img = cv2.resize(img, self.img_size)
        # targets = normalization_coordinate(targets, img_shape=img.shape)
        # target = constrain_coordinate(0, 1, targets)
        # return
        pass

    def collate_fn(self, batch):
        # imgs, targets = list(zip(*batch))
        # new_targets = []
        # for i, boxes in enumerate(targets):
        #     value = np.empty(boxes.shape[0])
        #     value.fill(i)
        #     new_targets.append(np.insert(boxes, 0, values=value, axis=1))
        # new_targets = torch.tensor(np.vstack(new_targets), dtype=torch.float32)
        # images = torch.tensor(imgs, dtype=torch.float32)
        # images = images.permute(0, 3, 1, 2).contiguous()
        # return images, new_targets
        pass


class PrepareDate(ImageFactory):
    def __init__(self):
        super(PrepareDate, self).__init__()

    def run(self, img, targets):
        """

        :param img:
        :param targets:
        :return:
        """
        # TODO Resize need to do
        img = img.astype(np.float32) / 255.0

        return img, targets


if __name__ == '__main__':
    # data_path = '~/data/object/voc/VOCdevkit/VOC2012'
    data_path = '~/datasets/VOC/VOCdevkit/VOC2007'
    data_set = ClassifyData(data_path)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(data_set, 32, shuffle=False, num_workers=1, collate_fn=data_set.collate_fn)
    for ii in range(2):
        for i, a in enumerate(dataloader):
            print(a)

    # a = np.array([[1, 2], [3, 4], [5, 6]])
    # np.random.shuffle(a)
    # print(123)
