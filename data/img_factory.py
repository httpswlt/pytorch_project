# coding:utf-8
"""
    This file contain the all handle functions of images.
"""
import random

import cv2
import numpy as np
from .utils import fip, rand


class ImageFactory:
    def __init__(self):
        self.rotate_rate = None
        self.zoom_rate = None
        self.jitter_x = None
        self.jitter_y = None
        self.hue = None
        self.saturation = None
        self.exposure = None
        self.img_size = None
        self.flip = None

    def __call__(self, *any):
        return self.run(*any)
    
    def set_image_size(self, img_size):
        """
        
        :param img_size: e.g.: (w, h)
        :return:
        """
        self.img_size = img_size
        
    def set_attribute(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters.get(parameter))
        assert self.img_size is not None, "Image Size can't be empty."

    def run(self, *any):
        raise NotImplementedError

    def random_jitter(self, img, targets=None):
        """

        :param img:
        :param targets: e.g.: [[cen_x, cen_y, w, h, cls],....], type: numpy.ndarray
        :return:    random jitter this img via the jitter_x and jitter_y
        """
        assert self.jitter_x is not None or self.jitter_y is not None
        if self.jitter_x is None:
            self.jitter_x = 0
        if self.jitter_y is None:
            self.jitter_y = 0

        img_height, img_width, img_channel = img.shape
        net_width = int(self.img_size[0])
        net_height = int(self.img_size[1])

        dw = self.jitter_x * img_width
        dh = self.jitter_y * img_height
        new_ar = (img_width + random.uniform(-dw, dw)) / (img_height + random.uniform(-dh, dh))

        scale = 1
        com_ratio = net_width / net_height
        if new_ar < com_ratio:
            nh = scale * net_height
            nw = nh * new_ar
        else:
            nw = scale * net_width
            nh = nw / new_ar
        im = cv2.resize(img, (int(nw), int(nh)))

        # coordinates operation
        if targets is not None:
            dw = int(nw) / img_width
            dh = int(nh) / img_height
            targets[:, :-1:2] *= dw
            targets[:, 1:-1:2] *= dh
            return im, targets

        return im

    def random_translation(self, img, targets=None):
        """

        :param img:
        :param targets: e.g.: [[cen_x, cen_y, w, h, cls],....], type: numpy.ndarray
        :return:    random translation this img shape to the input_size of net
        """
        net_width = self.img_size[0]
        net_height = self.img_size[1]
        nh, nw, img_channel = img.shape
        dx = random.uniform(0, net_width - nw)
        dy = random.uniform(0, net_height - nh)
        t_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, t_matrix, (net_width, net_height), flags=cv2.INTER_LINEAR,
                             borderValue=(0.5, 0.5, 0.5))

        if targets is not None:
            xy = targets[:, :2]
            xyz = np.hstack((xy, np.ones(xy.shape[0]).reshape(-1, 1)))
            xy = np.dot(xyz, t_matrix.T)
            targets[:, :2] = xy
            return img, targets
        return img

    def random_distort_image(self, img):
        """

        :param img: the value of image should range 0~1
        :return:
        """
        assert self.hue or self.saturation or self.exposure
        if self.hue is None:
            self.hue = 0
        if self.saturation is None:
            self.saturation = 1
        if self.exposure is None:
            self.exposure = 1

        hue = random.uniform(-self.hue, self.hue)

        saturation = random.uniform(1, self.saturation)
        saturation = saturation if rand() % 2 else 1. / saturation

        exposure = random.uniform(1, self.exposure)
        exposure = exposure if rand() % 2 else 1. / exposure

        img = self.distort_image(img, hue, saturation, exposure)
        return img

    @staticmethod
    def distort_image(img, hue, saturation, exposure):
        """

        :param img: the value of image should range 0~1
        :param hue:
        :param saturation:
        :param exposure:
        :return:
        """
        assert hue or saturation or exposure
        if hue is None:
            hue = 0
        if saturation is None:
            saturation = 1
        if exposure is None:
            exposure = 1

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
        img_hsv[:, :, 0] += hue  # hue
        img_hsv[:, :, 1] *= saturation  # saturation
        img_hsv[:, :, 2] *= exposure  # value
        img_hsv[:, :, 0][np.where(img_hsv[:, :, 0] > 1)] -= 1
        img_hsv[:, :, 0][np.where(img_hsv[:, :, 0] < 0)] += 1
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img

    def random_flip(self, img, targets=None, flip_code=[-1, 0, 1]):
        """

        :param img:
        :param flip_code:   must in the [-1, 0, 1].
        :param targets:
        :return:    random to flip image via the flip_code
        """
        assert self.flip is not None
        flip = rand() % 2
        if flip:
            flip_code = random.sample(flip_code, 1)[0]
            img = fip(img, flip_code)
            if targets is not None:
                nh, nw, img_channel = img.shape
                if flip_code == 1:
                    targets[:, 0] = nw - targets[:, 0]
                elif flip_code == 0:
                    targets[:, 1] = nh - targets[:, 1]
                elif flip_code == -1:
                    targets[:, 0] = nw - targets[:, 0]
                    targets[:, 1] = nh - targets[:, 1]
                return img, targets

        if targets is not None:
            return img, targets

        return img

    @staticmethod
    def pca_jitter(img):
        """

        :param img:
        :return:
        """

        img_size = img.size / 3
        print(img.size, img_size)
        img1 = img.reshape(int(img_size), 3)
        img1 = np.transpose(img1)
        img_cov = np.cov([img1[0], img1[1], img1[2]])

        lamda, p = np.linalg.eig(img_cov)

        p = np.transpose(p)

        alpha1 = random.normalvariate(0, 0.2)
        alpha2 = random.normalvariate(0, 0.2)
        alpha3 = random.normalvariate(0, 0.2)

        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
        add_num = np.dot(p, v)

        img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])

        img2 = np.swapaxes(img2, 0, 2)
        img2 = np.swapaxes(img2, 0, 1)

        return img2
