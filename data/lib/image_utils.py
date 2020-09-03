import random
import numpy as np


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
