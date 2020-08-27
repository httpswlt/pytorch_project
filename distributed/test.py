# coding:utf-8
# coding:utf-8
import sys

sys.path.append('../')
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import distributed, DataLoader
from torchvision.models.resnet import resnet18

from distributed.convert2apex import ConvertModel
from classify.datasets import load_imagenet_data
from classify.utils import validate
from backbone.darknet import DarknetClassify, darknet53


def run(gpu_id, config):
    cudnn.benchmark = True
    if config['distribute']:
        rank = config['rank'] * config['last_node_gpus'] + gpu_id
        gpu_id = gpu_id + config['start_node_gpus']
        print("world_size: {}, rank: {}, gpu: {}".format(config['world_size'], rank, gpu_id))
        dist.init_process_group(backend=config['backend'], init_method=config['ip'],
                                world_size=config['world_size'], rank=rank)
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.manual_seed(42)

    # create model
    model = DarknetClassify(darknet53())

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer strategy
    optimizer = torch.optim.SGD(model.parameters(), config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'])

    # define lr strategy
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=range(0, config['epochs'], config['epochs'] // 4),
                                                     gamma=0.5)

    if config.get('resume_path'):
        loc = 'cuda:{}'.format(gpu_id)
        checkpoint = torch.load(config.get('resume_path'), map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])

    # convert pytorch to apex model.
    parallel = DistributeModel(model, criterion, optimizer, config, gpu_id)
    parallel.convert()

    # load data
    train_sets, val_sets = load_imagenet_data(config['data_path'], )
    train_sampler = None

    if config['distribute']:
        train_sampler = distributed.DistributedSampler(train_sets)

    train_loader = DataLoader(train_sets, config['batch_size'], shuffle=(train_sampler is None),
                              num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_sets, config['batch_size'], shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True)
    dist.barrier()

    best_acc1 = 0
    # start training
    for epoch in range(config['epochs']):
        if config['distribute']:
            train_sampler.set_epoch(epoch)

        loss = parallel.train(epoch, train_loader)
        dist.barrier()
        lr = parallel.get_lr()
        if rank == 0:
            # train for per epoch
            print('Epoch: [{}/{}], Lr: {:.8f}'.format(epoch, config['epochs'], lr))
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, gpu_id)
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
        scheduler.step(loss)
        dist.barrier()

    dist.destroy_process_group()


def main():
    # ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 5
    config = {
        'ip': 'tcp://127.0.0.1:9999',
        'rank': 1,
        'last_node_gpus': 5,
        'world_size': 10,
        'start_node_gpus': 5,
        'backend': 'nccl',
        'distribute': True,
        'sync_bn': True,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 64,
        'num_workers': 4,
        'epochs': 160,
        'warmup_epoch': 0,
        'training_size': 256,
        'burn_in': 2000,

        'record': True,
        "model_path": "./model_lars_{}_{}_{}.checkpoint",
        "data_path": "/home/lintaowx/datasets/ImageNet2012",
        "resume_path": ""

    }
    mp.spawn(run, nprocs=ngpus_per_node, args=(config,))


if __name__ == '__main__':
    main()
