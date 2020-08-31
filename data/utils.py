# --------------------------------------------------------
# Project Name
# Copyright (c) Company
# Create by Lintao
# --------------------------------------------------------
import cv2
import random


def fip(img, flip_code):
    return cv2.flip(img, flipCode=flip_code)


def rand():
    return int(random.random() * 32767)
