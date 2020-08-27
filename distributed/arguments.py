# coding:utf-8
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', dest='ip', type=str, default='tcp://127.0.0.1:9999',
                        help='the ip address of main machine')
    parser.add_argument('-rank', '--rank', dest='rank', type=int, default=0,
                        help='which node, default: 0, meaning that it is the main machine')
    parser.add_argument('-lng', '--last_node_gpus', dest='last_node_gpus', type=int, default=0,
                        help='the gpu numbers of last machine node.')
    parser.add_argument('-ws', '--world_size', dest='world_size', type=int, default=10,
                        help='the total gpus numbers')
    parser.add_argument('-sng', '--start_node_gpus', dest='start_node_gpus', type=int, default=0,
                        help='start training on which gpu')
    parser.add_argument('-bk', '--backend', dest='backend', type=str, default='nccl',
                        help='the backbend of distributed')
    parser.add_argument('-syb', '--sync_bn', dest='sync_bn', action='store_true',
                        help='enable/disable synchronize batchnormal operate.')

    parser.add_argument('-lr', '--lr', dest='lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-momentum', '--momentum', dest='momentum', type=float, default=0.9,
                        help='the momentum of weights')
    parser.add_argument('-weight_decay', '--weight_decay', dest='weight_decay', type=float, default=0.0005,
                        help='the decay of weights')
    parser.add_argument('-epochs', '--epochs', dest='epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('-bs', '--batch-size', dest='batch-size', type=int, default=64,
                        help='size of each image batch')
    parser.add_argument('-dp', '--data_path', dest='data_path', type=str,
                        default="/home/lintaowx/datasets/ImageNet2012",
                        help='the path of training datasets')
    parser.add_argument('-nw', '--num_workers', dest='num_workers', type=int, default=4,
                        help='the numbers of datasets job threads')

    parser.add_argument('-bi', '--burn_in', dest='burn_in', type=int, default=2000,
                        help='the step numbers of lr burn_in')
    
    parser.add_argument('-sp', '--save_path', dest='save_path', type=str, default="./model_lars_{}_{}_{}.checkpoint",
                        help='the path of save model')
    parser.add_argument('-rd', '--record', dest='record', action='store_true',
                        help='whether record training process.')
    parser.add_argument('-rp', '--resume_path', dest='resume_path', type=str, default="",
                        help='the path of resume model')
    args = parser.parse_args()
    return args


def args2dict(args):
    return vars(args)


def dict2args(dic):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    return AttrDict(**dic)


if __name__ == '__main__':
    args = parse_args()
    dic = args2dict(args)
    args1 = dict2args(dic)
