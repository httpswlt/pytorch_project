# coding:utf-8
import time

from classify.utils import accuracy, AverageMeter, ProgressMeter
from training import BaseTraining


class Classifier(BaseTraining):
    def __init__(self):
        self.model = None
        super(Classifier, self).__init__()
    
    def set_model(self, model):
        self.model = model
    
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
        
        loss = 0
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.cuda:
                images = images.cuda()
                target = target.cuda()
            
            # compute output
            output = self.model(images)
            loss = self.criteria(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                print("Epoch {}, Steps: {}".format(epoch, i), end=" ")
                progress.display(i)
        return loss
