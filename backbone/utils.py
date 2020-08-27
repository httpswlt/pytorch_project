# coding:utf-8
from torch import nn


def activation_F(act='relu'):
    """

    :param act: you can define which Activate Function to use.
    :return:
    """
    if act == 'relu':
        act = nn.ReLU(inplace=True)
    elif act == 'leaky':
        act = nn.LeakyReLU(0.1, inplace=True)
    else:
        raise RuntimeError("the {} activation function not implement.".format(act))
    return act


def con_bn_act(in_channel, out_channel, kernel_size, stride, padding=0, bn=True, act='relu', bias=True, convert=True):
    """

    :param in_channel: input channels
    :param out_channel: output channels
    :param kernel_size: as the convolution operate
    :param stride: as the convolution operate
    :param padding: as the convolution operate  default: 0
    :param bn: whether enable batchnormal operate   default: True(enable)
    :param act: it only support relu, leaky and linear operate. default: relu.
    :param bias: as the convolution operate, default: False.
    :param convert: whether wrap the list via torch.nn.Sequential
    :return: convolution + bn + activate
    """
    layers = list()
    pad = (kernel_size - 1) // 2 if padding else 0
    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channel))

    if act != 'linear':
        layers.append(activation_F(act))

    if convert:
        layers = nn.Sequential(*layers)
    return layers
