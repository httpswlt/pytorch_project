# coding:utf-8

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import random
import warnings
from torch.utils.data import distributed, DataLoader
from torchvision.models.resnet import resnet18

import sys

sys.path.append('../')


# from distributed.convert2apex import ConvertModel
# from classify.datasets import load_imagenet_data
# from classify.utils import validate
# from backbone.darknet import DarknetClassify, darknet53


class DistributedTraining:
    def __init__(self, model, config):
        """

        :param config:
        """
        super(DistributedTraining, self).__init__()
        self.model = model
        self.config = config

    def __call__(self, *args, **kwargs):
        mp.spawn(self.run, nprocs=self.config.ngpus_per_node, args=())

    def run(self, gpu_id):
        cudnn.benchmark = True
        self.fix_seed(self.config.seed)
        if self.config.distribute:
            self.multi_node_multi_gpu(gpu_id)
        else:
            pass

    def single_node_multi_gpu(self):
        pass

    def multi_node_multi_gpu(self, gpu_id):
        model = self.model
        rank = self.config.node * self.config.last_node_gpus + gpu_id
        gpu_id = gpu_id + self.config.start_node_gpus
        print("world_size: {}, rank: {}, gpu: {}".format(self.config.world_size, rank, gpu_id))
        assert rank < self.config.world_size
        dist.init_process_group(backend=self.config.backend, init_method=self.config.ip,
                                world_size=self.config.world_size, rank=rank)
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)
        model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])


    @staticmethod
    def fix_seed(seed):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')


if __name__ == '__main__':
    print(123)
