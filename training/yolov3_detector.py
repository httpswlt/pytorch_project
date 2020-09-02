# coding:utf-8
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from test.yolov3_test import test
from training import Detector
from numba import jit


class Yolov3Detecor(Detector):
    def __init__(self):
        super(Yolov3Detecor, self).__init__()
        self.best_loss = float('inf')
        self.best_map_iou80 = float(0)
        self.best_map_iou50 = float(0)
        self.best_pr_iou80 = [0., 0.]
        self.best_pr_iou50 = [0., 0.]
        self.results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
        self.mloss = np.zeros(5)
        self.burn_in = 1000
        self.accumulate = 1
        self.start_epoch = 0
        self.device_ids = None
        
        self.ckpt = None
        self.device = None
        self.batch_size = 0
        self.tensorboadlog = None
    
    def set_device_ids(self, device_ids=None):
        self.device_ids = device_ids
    
    def train(self):
        print(self)
        nb = len(self.dataloader)
        
        # Start training
        t = time.time()
        results = (0, 0, 0, 0, 0)
        burnin_count = int(self.start_epoch * len(self.dataloader) / self.accumulate)
        
        for epoch in range(self.start_epoch, self.epochs):
            print(
                ('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
            
            print('learning rate: %g' % self.optimizer.param_groups[0]['lr'])
            
            self.mloss = np.zeros(5)
            for i, (img, target) in enumerate(self.dataloader):
                img = img.to(self.device)
                target = target.to(self.device)
                target_number = len(target)
                
                # SGD burn-in
                if self.burn_in and burnin_count < self.burn_in:
                    self.burn_in_func(burnin_count)
                    burnin_count += 1
                
                # compute the network loss
                pred, loss, loss_items = self.inference(img, target)
                loss = torch.mean(loss)
                loss_items = torch.mean(loss_items.view((-1, loss_items.size()[0])), 0)
                
                if torch.isnan(loss):
                    print('WARNING: nan loss detected, ending training')
                    return
                
                # Compute gradient
                loss.backward()
                # Accumulate gradient for x batches before optimizing
                if (i + 1) % self.accumulate == 0 or (i + 1) == nb:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update running mean of tracked metrics
                self.mloss = (self.mloss * i + loss_items.data.cpu().numpy()) / (i + 1)
                
                # Print batch results
                s = ('%8s%12s' + '%10.3g' * 7) % (
                    '%g/%g' % (epoch, self.epochs - 1), '%g/%g' % (i, nb - 1), *self.mloss, target_number,
                    time.time() - t)
                t = time.time()
                print(s)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.save_log(s, self.mloss, epoch)
    
    def save_log(self, message, mloss, epoch):
        self.tensorboadlog = 'log'
        if not self.tensorboadlog:
            return
        
        writer = SummaryWriter(self.tensorboadlog)
        writer.add_scalar('loss/total_loss', mloss[0], epoch)
        writer.add_scalar('loss/lxy', mloss[1], epoch)
        writer.add_scalar('loss/lwh', mloss[2], epoch)
        writer.add_scalar('loss/lconf', mloss[3], epoch)
        writer.add_scalar('loss/lcls', mloss[4], epoch)
        
        with torch.no_grad():
            results, maps = test(self)
            # Tensorboard for test
            writer.add_scalar('eval/precision', results[0], epoch)
            writer.add_scalar('eval/recall', results[1], epoch)
            writer.add_scalar('eval/ap', results[2], epoch)
            writer.add_scalar('eval/f1', results[3], epoch)
            writer.add_scalar('eval/test_loss', results[4], epoch)
            writer.add_scalar('eval_IOU50/precision', results[5], epoch)
            writer.add_scalar('eval_IOU50/recall', results[6], epoch)
            writer.add_scalar('eval_IOU50/ap', results[7], epoch)
            writer.add_scalar('eval_IOU50/f1', results[8], epoch)
        
        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(message + '%11.3g' * 9 % results + '\n')  # P, R, mAP, F1, test_loss
        
        best_loss = float('inf')
        best_map_iou80 = float(0)
        best_map_iou50 = float(0)
        best_pr_iou80 = [0., 0.]
        best_pr_iou50 = [0., 0.]
        weights = 'weights' + os.sep
        latest = weights + 'latest.pt'
        best = weights + 'best.pt'
        best_map_path = weights + 'best_map.pt'
        best_pr_path = os.path.join(weights, 'best_pr.pt')
        if not os.path.exists(weights):
            os.makedirs(weights)
        
        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss
        
        # Update best map
        test_map_iou80 = results[2]
        test_map_iou50 = results[7]
        if test_map_iou80 >= best_map_iou80 and test_map_iou50 >= best_map_iou50:
            best_map_iou80 = test_map_iou80
            best_map_iou50 = test_map_iou50
        
        test_pr_iou80 = results[0], results[1]
        test_pr_iou50 = results[5], results[6]
        if test_pr_iou80[0] >= best_pr_iou80[0] and test_pr_iou80[1] >= best_pr_iou80[1] and \
                test_pr_iou50[0] >= best_pr_iou50[0] and test_pr_iou50[1] >= best_pr_iou50[1] and \
                test_pr_iou80[0] > 70. and test_pr_iou80[1] > 70. and test_pr_iou50[0] > 90. and test_pr_iou50[1] > 90.:
            best_pr_iou80 = test_pr_iou80
            best_pr_iou50 = test_pr_iou50
        
        # Save training results
        # Create checkpoint
        chkpt = {'epoch': epoch,
                 'best_loss': best_loss,
                 'best_map': best_map_iou80,
                 'best_pr': best_pr_iou80,
                 'model': self.backbone.module.state_dict() if type(
                     self.backbone) is nn.parallel.DistributedDataParallel else self.backbone.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        
        # Save latest checkpoint
        torch.save(chkpt, latest)
        # Save best checkpoint
        if best_loss == test_loss:
            torch.save(chkpt, best)
        # Save best map checkpoint
        if best_map_iou80 == test_map_iou80:
            torch.save(chkpt, best_map_path)
        # Save best pr checkpoint
        if best_pr_iou80[0] == test_pr_iou80[0] and best_pr_iou80[1] == test_pr_iou80[1] and \
                best_pr_iou50[0] == test_pr_iou50[0] and best_pr_iou50[1] == test_pr_iou50[1]:
            torch.save(chkpt, best_pr_path)
        
        # Save better model
        if test_pr_iou80[0] > 72. and test_pr_iou80[1] > 70. and test_pr_iou50[0] > 99.3 and test_pr_iou50[1] > 95.:
            torch.save(chkpt, weights + 'backup_target_pr%g.pt' % epoch)
        
        # Save target model
        if test_pr_iou80[0] > 74.34 and test_pr_iou80[1] > 71.33 and test_pr_iou50[0] > 99.54 and test_pr_iou50[
            1] > 95.5:
            torch.save(chkpt, weights + 'backup_better_pr%g.pt' % epoch)
        
        # Save backup every 10 epochs (optional)
        if epoch > 0 and epoch % 20 == 0:
            torch.save(chkpt, weights + 'backup%g.pt' % epoch)
        
        # Delete checkpoint
        del chkpt
        writer.close()
    
    def inference(self, img, target):
        loss = 0
        lxy = 0
        lwh = 0
        lconf = 0
        lcls = 0
        detections = []
        features = self.backbone(img)
        for j, feature in enumerate(features):
            self.header.set_anchors(self.anchors[j])
            x, layer_loss, layer_lxy, layer_lwh, layer_lconf, layer_lcls = self.header((feature, target))
            loss += layer_loss
            lxy += layer_lxy
            lwh += layer_lwh
            lconf += layer_lconf
            lcls += layer_lcls
            detections.append(x)
        detections = torch.cat(detections, dim=1)
        if target is not None:
            total_numbers = len(features) * feature.shape[0]
            loss = loss / total_numbers
            lxy, lwh = lxy / total_numbers, lwh / total_numbers
            lconf, lcls = lconf / total_numbers, lcls / total_numbers
            return detections, loss, torch.cat((loss, lxy, lwh, lconf, lcls)).detach()
        return detections
    
    def burn_in_func(self, count):
        if count <= self.burn_in:
            lr = self.lr * (count / self.burn_in) ** 4
            for x in self.optimizer.param_groups:
                x['lr'] = lr
    
    def set_hyper_parameters(self, hyper_parameters):
        # attributes = self.__dict__.keys()
        for parameter in hyper_parameters:
            setattr(self, parameter, hyper_parameters.get(parameter))
        
        self.device = self.get_device_info(self.cuda, self.device_ids)
        self.backbone.to(self.device)
        if self.cuda:
            # multi GPU to training network
            if torch.cuda.device_count() > 1:
                self.backbone = nn.DataParallel(self.backbone, device_ids=self.device_ids)
        
        # Load previously saved model(resume from ***.pt)
        if self.ckpt:
            self.load_weights(self.ckpt)
        
        self.batch_size = self.dataloader.batch_size
    
    def load_weights(self, weight_file):
        name, ftyple = os.path.splitext(weight_file)
        if ftyple not in ['pkl', 'pt', 'pth']:
            with open(weight_file, "rb") as fp:
                weights = np.fromfile(fp, dtype=np.float32)  # The rest of the values are the weights
                modules = self.get_all_children_modules(self.backbone.modules())
                ptr = 0
                for i, module in enumerate(modules):
                    if isinstance(module, nn.Conv2d):
                        bn = modules[i + 1]
                        if isinstance(bn, nn.BatchNorm2d):
                            num_bn_biases = bn.bias.numel()
                            # Load the weights
                            bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                            ptr += num_bn_biases
                            bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr += num_bn_biases
                            bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr += num_bn_biases
                            bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr += num_bn_biases
                            # Cast the loaded weights into dims of model weights.
                            bn_biases = bn_biases.view_as(bn.bias.data)
                            bn_weights = bn_weights.view_as(bn.weight.data)
                            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                            bn_running_var = bn_running_var.view_as(bn.running_var)
                            # Copy the data to model
                            bn.bias.data.copy_(bn_biases)
                            bn.weight.data.copy_(bn_weights)
                            bn.running_mean.copy_(bn_running_mean)
                            bn.running_var.copy_(bn_running_var)
                        else:
                            # Number of biases
                            num_biases = module.bias.numel()
                            # Load the weights
                            conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                            ptr = ptr + num_biases
                            # reshape the loaded weights according to the dims of the model weights
                            conv_biases = conv_biases.view_as(module.bias.data)
                            # Finally copy the data
                            module.bias.data.copy_(conv_biases)
                        # Let us load the weights for the Convolutional layers
                        num_weights = module.weight.numel()
                        # Do the same as above for weights
                        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                        ptr = ptr + num_weights
                        conv_weights = conv_weights.view_as(module.weight.data)
                        module.weight.data.copy_(conv_weights)
        else:
            chkpt = torch.load(weight_file, map_location=self.device)  # load checkpoint
            self.backbone.load_state_dict(chkpt['model'])
            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
