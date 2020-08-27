import time

import apex
from classify.utils import accuracy, AverageMeter, ProgressMeter
from distributed.lars import LARS
import torch
from classify.utils import gradual_warmup, adjust_learning_rate_step, adjust_learning_rate_poly


class ConvertModel(object):
    def __init__(self, model, criterion, optimizer, config, gpu_id=None):
        super(ConvertModel, self).__init__()
        self.model = model
        self.config = config
        self.sync_bn = self.config['sync_bn']
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            assert isinstance(self.gpu_id, int), "GPU should is a int type."
        self.criterion = criterion
        self.optimizer = optimizer
        self.opt_level = None
        self.max_epoch = config['epochs']
        self.burn_in = config['burn_in']
        self.burnin_count = 0

    def convert(self, opt_level=None):
        self.opt_level = opt_level
        torch.cuda.set_device(self.gpu_id)

        # assign specific gpu
        self.model = self.model.cuda(self.gpu_id)
        self.criterion = self.criterion.cuda(self.gpu_id)

        if self.opt_level is None:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])
        else:
            if self.sync_bn:
                # synchronization batch normal
                self.model = apex.parallel.convert_syncbn_model(self.model)
            # init model and optimizer by apex
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.opt_level)
            # apex parallel
            self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)
        return self.model, self.criterion, self.optimizer

    def lars(self):
        self.optimizer = LARS(self.optimizer)

    def get_lr(self):
        return self.optimizer.param_groups[0].get('lr')

    def train(self, epoch, train_loader):
        """
        you must run it after the 'convert' function.
        :param epoch:
        :param train_loader:
        :return:
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.gpu_id is not None:
                images = images.cuda(self.gpu_id, non_blocking=True)
                target = target.cuda(self.gpu_id, non_blocking=True)

            # SGD burn-in
            if self.burnin_count < self.burn_in:
                self.burnin_count = i + len(train_loader) * epoch
                lr = self.config['lr'] * (self.burnin_count / self.burn_in) ** 4
                for x in self.optimizer.param_groups:
                    x['lr'] = lr

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()

            if self.opt_level is None:
                loss.backward()
            else:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                if self.gpu_id == self.config['start_node_gpus']:
                    print("Epoch {}, Steps: {}".format(epoch, i), end=" ")
                    progress.display(i)
        return loss
