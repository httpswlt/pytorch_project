# coding:utf-8
from torch import optim
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, './')

from backbone import Darknet
from common import read_yaml
from data import PrepareDate
from data import Yolov3Data
from header import Yolov3Header
from training import Yolov3Detecor

net_yaml = './config/net.yaml'
data_path = '/home/lintaowx/datasets/sports-training-data/player_detection/training_dataset'
net = read_yaml(net_yaml)
data = read_yaml(net['data_yaml'])

prepare = PrepareDate()
prepare.set_attribute(data)
data_set = Yolov3Data(data, prepare, index_file='train_freed_2k')
batch_size = 4
dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=0, collate_fn=data_set.collate_fn)

backbone = Yolov3BackBone(len(net['anchors'][0]) * (5 + net.get('classes_num')))
header1 = YOLOHeader(net.get('classes_num'), data.get('img_size'), net.get('ignore_thres_first_loss'))
optimzer = optim.SGD(backbone.parameters(), lr=net['lr'],
                     momentum=net.get('momentum', 0),
                     weight_decay=net.get('weight_decay', 0))
model = Yolov3Detecor()
model.set_device_ids((3,))
model.set_dataloader(dataloader)
model.set_backbone(backbone)
model.set_header(header1)
model.set_optimizer(optimzer)
model.set_hyper_parameters(net)
model.set_lr_scheduler(optim.lr_scheduler.MultiStepLR(model.optimizer,
                                                      milestones=range(model.start_epoch, model.epochs,
                                                                       int((model.epochs - model.start_epoch) / 3)),
                                                      gamma=0.1, last_epoch=model.start_epoch - 1
                                                      )
                       )

model.train()
