# coding:utf-8
import sys

sys.path.append('../')
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
# from classify.datasets import load_imagenet_data
from classify.utils import validate
from data.classify_data import ClassifyData, PrepareData
from backbone.darknet import DarknetClassify, darknet53
from training import Classifier


def run(config):
    cudnn.benchmark = True
    torch.manual_seed(42)
    
    # create model
    model = DarknetClassify(darknet53(), num_classes=3)
    
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
    classifier = Classifier()
    classifier.set_model(model)
    classifier.set_criteria(criterion)
    classifier.set_optimizer(optimizer)
    classifier.set_lr_scheduler(lr_scheduler)
    
    # load data
    prepare_data = PrepareData()
    prepare_data.set_image_size((600, 200))
    train_sets = ClassifyData(config['data_path'], prepare_data)
    
    train_loader = DataLoader(train_sets, config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True,
                              collate_fn=train_sets.collate_fn)
    # val_loader = DataLoader(val_sets, config['batch_size'], shuffle=False, num_workers=config['num_workers'],
    #                         pin_memory=True)
    
    best_acc1 = 0
    # start training
    for epoch in range(config['epochs']):
        
        loss = classifier.train(epoch, train_loader)
        lr = classifier.get_lr()
        
        # train for per epoch
        print('Epoch: [{}/{}], Lr: {:.8f}'.format(epoch, config['epochs'], lr))
        # evaluate on validation set
        acc1, acc5 = validate(train_loader, model, criterion)
        if config['record']:
            with open('record.log', 'a') as f:
                f.write('Epoch {}, lr {:.8f}, loss: {:.8f}, Acc@1 {:.8f}, Acc5@ {:.8f} \n'.
                        format(epoch, lr, loss, acc1, acc5))
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # remember best acc@1 and save checkpoint
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, config['model_path'].format(epoch, best_acc1, loss))
        lr_scheduler.step(epoch)
        dist.barrier()


def main():
    config = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 2,
        'num_workers': 4,
        'epochs': 160,
        'warmup_epoch': 0,
        'burn_in': 2000,
        
        'record': True,
        "model_path": "./model_lars_{}_{}_{}.checkpoint",
        "data_path": "/home/lintaowx/data/st/60",
        "resume_path": ""
        
    }
    run(config)


if __name__ == '__main__':
    
    main()
