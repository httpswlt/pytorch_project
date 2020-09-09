# coding: utf-8
from torchvision.models import vgg
import torch
import torch.nn as nn
from backbone import BaseBone
import torchbnn as bnn


class VGG(BaseBone):
    cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }
    
    def __init__(self, cfg='vgg16', batch_normal=False, bayes=False):
        super(VGG, self).__init__()
        assert cfg in self.cfgs.keys()
        
        self.net = self.cfgs[cfg]
        self.bayes = bayes
        self.batch_normal = batch_normal
        self.layers = self.make_layers()
        self.set_last_output_channels(self.net[-2])
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def make_layers(self):
        layers = []
        in_channels = 3
        for v in self.net:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.bayes:
                    conv2d = bnn.BayesConv2d(0, 0.1, in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_normal:
                    if self.bayes:
                        bn = bnn.BayesBatchNorm2d(0, 0.1, v)
                    else:
                        bn = nn.BatchNorm2d(v)
                    layers += [conv2d, bn, nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class VGGClassifier(BaseBone):
    def __init__(self, backbone, num_classes=1000, bayes=False):
        super(VGGClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(backbone.get_last_output_channels() * 7 * 7, 4096)
            if not bayes else bnn.BayesLinear(0, 0.1, backbone.get_last_output_channels() * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096)
            if not bayes else bnn.BayesLinear(0, 0.1, 4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
            if not bayes else bnn.BayesLinear(0, 0.1, 4096, num_classes)
        )
        self.initialize_weights_kaiming()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    import os
    
    # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '9')
    # np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(123)  # cpu
    torch.cuda.manual_seed_all(123)  # gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu
    torch.backends.cudnn.benchmark = True
    bayes = True
    inputs = torch.ones((1, 3, 256, 256)).cuda()
    model = VGG('vgg16', True, bayes).cuda()
    # x = model(inputs)
    vgg = VGGClassifier(model, 3, bayes).cuda()
    x = vgg(inputs)
    print(x)
    # darknet.initialize_weights()
