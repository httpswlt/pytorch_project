# coding: utf-8
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--debug', action='store_true', help='debug training')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--notensorboard', action='store_true', help='use tensorboard monitor')
    parser.add_argument('--save_log', type=str, default='log', help='tensorboard save log path')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    parser.add_argument('--init_weights', type=str, default='yolov3-player_stage2_start.81',
                        help='initialization model weights file name')
    parser.add_argument('--weights', type=str, default='weights', help='weights file path')

    opt = parser.parse_args()
