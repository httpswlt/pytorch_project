# coding:utf-8
import torch
from torch import nn

from backbone import BaseBone, con_bn_act


class Yolov3Header(BaseBone):
    def __init__(self, in_channels, out_channels, backbone):
        """

        :param in_channels: the channels of backbone's output
        :param out_channels: outputs channels
        :param backbone: backbone network
        """
        super(Yolov3Header, self).__init__()
        self.planes = out_channels
        self.backbone = backbone
        self.out_channel = in_channels

        # yolo layer1 construct
        out_channel = self.out_channel
        self.yolo1 = self._make_layers(out_channel, 1)

        # yolo layer2 construct
        out_channel = int(out_channel / 2)
        self.conv2_1 = con_bn_act(in_channel=out_channel, out_channel=256, kernel_size=1,
                                  stride=1, padding=1, act='leaky')
        self.upsample2_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo2 = self._make_layers(int(out_channel / 2 + out_channel), 2)

        # yolo layer3 construct
        out_channel = int(out_channel / 2)
        self.conv3_1 = con_bn_act(in_channel=out_channel, out_channel=128, kernel_size=1,
                                  stride=1, padding=1, act='leaky')
        self.upsample3_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo3 = self._make_layers(int(out_channel / 2 + out_channel), 3)

    def forward(self, x):
        backbone_x = self.backbone(x)
        x = backbone_x
        # yolo1 outputs
        yolo1_outputs = []
        for layer in self.yolo1:
            x = layer(x)
            yolo1_outputs.append(x)

        # yolo2 outpus
        x = yolo1_outputs[-3]
        x = self.conv2_1(x)
        x = self.upsample2_1(x)
        x = torch.cat((x, self.backbone.features[-2]), 1)
        yolo2_outputs = []
        for layer in self.yolo2:
            x = layer(x)
            yolo2_outputs.append(x)

        # yolo3 outpus
        x = yolo2_outputs[-3]
        x = self.conv3_1(x)
        x = self.upsample3_1(x)
        x = torch.cat((x, self.backbone.features[-3]), 1)
        yolo3_outputs = []
        for layer in self.yolo3:
            x = layer(x)
            yolo3_outputs.append(x)
        return yolo1_outputs[-1], yolo2_outputs[-1], yolo3_outputs[-1]

    def _make_layers(self, in_channel, branch_i):
        layers = []
        temp_channel = int(self.out_channel / (2 ** branch_i))
        for i in range(3):
            layers.append(
                con_bn_act(in_channel=in_channel, out_channel=temp_channel, kernel_size=1,
                           stride=1, padding=1, act='leaky'))
            layers.append(
                con_bn_act(in_channel=temp_channel, out_channel=temp_channel * 2, kernel_size=3,
                           stride=1, padding=1, act='leaky'))
            in_channel = temp_channel * 2

        layers.append(con_bn_act(in_channel=temp_channel * 2, out_channel=self.planes, kernel_size=1,
                                 stride=1, padding=1, act='linear'))
        return nn.ModuleList(layers)


if __name__ == '__main__':
    from backbone.darknet import darknet53

    mask = [6, 7, 8]
    classes = 1
    inputs = torch.ones((1, 3, 256, 256)).cuda()
    backbone = darknet53()
    darknet = Yolov3Header(backbone.get_last_output_channels(), len(mask) * (5 + classes), backbone).cuda()
    # inference = darknet(inputs)
    # from torch2trt import torch2trt
    #
    # model = torch2trt(darknet, [inputs])
    # import time
    #
    # time1 = time.time()
    # a = darknet(inputs)
    # time2 = time.time()
    # b = model(inputs)
    # time3 = time.time()
    # print('Original time: ', time2 - time1)
    # print('TensorRT time: ', time3 - time2)
    # print(torch.max(torch.abs(b - a)))
