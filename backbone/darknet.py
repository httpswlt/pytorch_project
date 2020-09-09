# coding:utf-8
from torch import nn

from backbone import BaseBone, con_bn_act


class Darknet(BaseBone):
    def __init__(self, bottleneck, blocks, in_channel=3):
        super(Darknet, self).__init__()
        self.bottleneck = bottleneck
        self.conv1 = con_bn_act(in_channel=in_channel, out_channel=32,
                                kernel_size=3, stride=1, padding=1, bn=True, act='leaky')
        self.layers = []
        for i, block in enumerate(blocks):
            i = 2 ** i if i else 1
            self.layers.append(self.__make_layer(32 * i, 64 * i, block))
        self.layers = nn.ModuleList(self.layers)
        self.features = []
        self.set_last_output_channels(64 * i)

    def forward(self, x):
        self.features = []
        x = self.conv1(x)
        self.features.append(x)

        for layer in self.layers:
            x = layer(x)
            self.features.append(x)

        return x

    def __make_layer(self, in_channel, out_channel, block):
        layers = list()
        layers.append(con_bn_act(in_channel, out_channel, kernel_size=3, stride=2, padding=1,
                                 bn=True, act='leaky'))
        for _ in range(block):
            layers.append(self.bottleneck(in_channel, out_channel))
        return nn.Sequential(*layers)


class Shortcut(BaseBone):
    def __init__(self):
        super(Shortcut, self).__init__()

    def forward(self, x, y):
        return x + y


class Bottleneck(BaseBone):
    def __init__(self, in_channel, out_channel):
        super(Bottleneck, self).__init__()

        self.conv1 = con_bn_act(in_channel=out_channel, out_channel=in_channel, kernel_size=1,
                                stride=1, padding=1, bn=True, act='leaky')

        self.conv2 = con_bn_act(in_channel=in_channel, out_channel=out_channel, kernel_size=3,
                                stride=1, padding=1, bn=True, act='leaky')

        self.shortcut = Shortcut()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.shortcut(x, out)
        return out


class DarknetClassifier(BaseBone):
    def __init__(self, backbone, num_classes=1000):
        super(DarknetClassifier, self).__init__()
        self.num_classes = num_classes

        self.backbone = backbone
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.backbone.get_last_output_channels(), self.num_classes)
        self.initialize_weights_xavier_normal()

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def darknet53():
    return Darknet(Bottleneck, [1, 2, 8, 8, 4])


if __name__ == '__main__':
    import torch

    torch.manual_seed(123)  # cpu
    torch.cuda.manual_seed_all(123)  # gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu
    torch.backends.cudnn.benchmark = True

    inputs = torch.ones((1, 3, 256, 256)).cuda()
    model = darknet53().cuda()
    model(inputs)
    darknet = DarknetClassifier(model).cuda()
    print(darknet)
