# --------------------------------------------------------
# Project Name
# Copyright (c) Company
# Create by Lintao
# --------------------------------------------------------
import sys
import os

dir_path = os.path.abspath(__file__)
pro_path = os.path.abspath(os.path.join(dir_path, '..', '..', '..'))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
if pro_path not in sys.path:
    sys.path.insert(0, pro_path)
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import os

# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1, 2, 3, 4, 5, 6, 7, 8, 9')
# from classify.datasets import load_imagenet_data
from data.classify_data import ClassifyData, PrepareData
from backbone.vgg import VGG, VGGClassifier
from training import BayesClassifierTraining


def run(config):
    cudnn.benchmark = True
    torch.manual_seed(42)
    # create some path
    os.makedirs(config['model_path'], exist_ok=True)

    # create model
    backbone = VGG('vgg16', batch_normal=True, bayes=True)
    model = VGGClassifier(backbone=backbone, num_classes=config['classes_num'], bayes=True).cuda()

    model = torch.nn.parallel.DataParallel(model)
    print(model)
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])

    # define lr strategy
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=range(0, config['epochs'], config['epochs'] // 4),
                                                        gamma=0.5)

    if config.get('resume_path'):
        checkpoint = torch.load(config.get('resume_path'))
        model.load_state_dict(checkpoint['state_dict'])

    # set the trainer
    classifier = BayesClassifierTraining()
    classifier.set_model(model)
    classifier.set_criteria(criterion)
    classifier.set_optimizer(optimizer)
    classifier.set_lr_scheduler(lr_scheduler)

    # load data
    prepare_data = PrepareData()
    prepare_data.set_image_size(config['image_size'])
    train_sets = ClassifyData(os.path.join(config['data_path'], 'train'), prepare_data)
    val_sets = ClassifyData(os.path.join(config['data_path'], 'val'), prepare_data)

    train_loader = DataLoader(train_sets, config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True,
                              collate_fn=train_sets.collate_fn)
    val_loader = DataLoader(val_sets, config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True,
                            collate_fn=train_sets.collate_fn)

    best_acc1 = 0
    # start training
    for epoch in range(config['epochs']):

        loss = classifier.train(epoch, train_loader)
        lr = classifier.get_lr()

        # train for per epoch
        print('Epoch: [{}/{}], Lr: {:.8f}'.format(epoch, config['epochs'], lr))
        # evaluate on validation set
        acc1 = classifier.validate(val_loader)
        if config['record']:
            with open('record.log', 'a') as f:
                f.write('Epoch {}, lr {:.8f}, loss: {:.8f}, Acc@1 {:.8f}\n'.
                        format(epoch, lr, loss.item(), acc1.item()))
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # remember best acc@1 and save checkpoint
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(config['model_path'],
                            'model_{}_{}_{}.checkpoint'.format(epoch, best_acc1.item(), loss.item())))
        lr_scheduler.step()


def main():
    config = {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 12,
        'num_workers': 0,
        'epochs': 200,

        'image_size': (500, 170),
        'classes_num': 2,

        'record': True,
        "model_path": "./weights",
        "data_path": "/home/lintaowx/data/st_middle/120",
        "resume_path": ""

    }
    print(config)
    run(config)


if __name__ == '__main__':
    main()
