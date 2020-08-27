# coding:utf-8
from torch import nn
from functools import partial


class BaseBone(nn.Module):
    """
        the Base class of all backbone network.
    """
    def __init__(self):
        super(BaseBone, self).__init__()
        self.__last_output_channels = None

    def set_last_output_channels(self, channels):
        self.__last_output_channels = channels

    def get_last_output_channels(self):
        assert self.__last_output_channels is not None
        return self.__last_output_channels

    def initialize_weights_kaiming(self):
        func = partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')
        self.item_init_modules(self._modules, func)

    def initialize_weights_xavier_normal(self):
        self.item_init_modules(self._modules, nn.init.xavier_normal_)

    def item_init_modules(self, modules, func):
        for _, module in modules.items():
            module_next = module._modules
            if len(module_next) == 0:
                if isinstance(module, nn.Conv2d):
                    func(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)
            else:
                self.item_init_modules(module_next, func)
