# coding:utf-8
import numpy as np

from data import ImageFactory


class PrepareDate(ImageFactory):
    def __init__(self):
        super(PrepareDate, self).__init__()

    def run(self, img, targets):
        """

        :param img:
        :param targets:
        :return:
        """

        img = img.astype(np.float32) / 255.0
        img, targets = self.random_jitter(img, targets)
        img = self.random_distort_image(img)
        img, targets = self.random_translation(img, targets)
        img, targets = self.random_flip(img, targets)
        return img, targets
