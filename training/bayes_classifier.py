# coding:utf-8
import time
import torch

from classify.utils import accuracy, AverageMeter, ProgressMeter
from training import ClassifierTraining
import torchbnn as bnn
import numpy as np


class BayesClassifierTraining(ClassifierTraining):
    def __init__(self):
        super(BayesClassifierTraining, self).__init__()
        self.model = None
        self.kl_weight = 0.1
        self.kl_loss = bnn.BKLLoss()

    def set_model(self, model):
        self.model = model

    def set_kl_weight(self, kl_weight):
        self.kl_weight = kl_weight

    def set_kl_loss(self, kl_loss):
        self.kl_loss = kl_loss

    def _step_update(self, images, target):
        # compute output
        output = self.model(images)
        classify_loss = self.criteria(output, target)
        kl_loss = self.kl_loss(self.model)
        loss = classify_loss + self.kl_weight * kl_loss

        return output, classify_loss, kl_loss, loss

    def train(self, epoch, train_loader):
        """
        you must run it after the 'convert' function.
        :param epoch:
        :param train_loader:
        :return:
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        classify_losses = AverageMeter('Classify_Loss', ':.4e')
        kl_losses = AverageMeter('KL_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, classify_losses, kl_losses, top1],
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

            output, classify_loss, kl_loss, loss = self._step_update(images, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))[0]
            classify_losses.update(classify_loss.item(), images.size(0))
            kl_losses.update(kl_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

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

    def validate(self, val_loader):
        batch_time = AverageMeter('Time', ':6.3f')
        classify_losses = AverageMeter('Classify_Loss', ':.4e')
        kl_losses = AverageMeter('KL_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        alea = AverageMeter('Alea', ':6.2f')
        epis = AverageMeter('Epis', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, classify_losses, kl_losses, top1, alea, epis],
            prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if self.cuda:
                    images = images.cuda()
                    target = target.cuda()
                output, classify_loss, kl_loss, loss = self._step_update(images, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))[0]
                classify_losses.update(classify_loss.item(), images.size(0))
                kl_losses.update(kl_loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

                # measure the uncertainty
                preds, aleatoric, epistemic = self.compute_uncertainty(images)
                alea.update(aleatoric, images.size(0))
                epis.update(epistemic, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    progress.display(i)

            print(' * Acc@1 {top1.avg:.3f}, Aleatoric {alea.avg:.3f }, Epistemic {epis.avg:.3f}'
                  .format(top1=top1, alea=alea, epis=epis))

        return top1.avg, alea.avg, epis.avg

    def compute_uncertainty(self, images, normalized=True, t=15):
        batch_predictions = []
        net_outs = []
        images = images.unsqueeze(0).repeat(t, 1, 1, 1, 1)

        preds = []
        epistemics = []
        aleatorics = []

        for i in range(t):  # for t batches
            net_out = self.model(images[i].cuda())
            net_outs.append(net_out)
            if normalized:
                prediction = torch.nn.functional.softplus(net_out)
                prediction = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
            else:
                prediction = torch.nn.functional.softmax(net_out, dim=1)
            batch_predictions.append(prediction)

        for sample in range(images.shape[0]):
            # for each sample in a batch
            pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs], dim=0)
            pred = torch.mean(pred, dim=0)
            preds.append(pred)

            p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions],
                              dim=0).detach().cpu().numpy()
            p_bar = np.mean(p_hat, axis=0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic = np.dot(temp.T, temp) / t
            epistemic = np.diag(epistemic)
            epistemics.append(epistemic)

            aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / t)
            aleatoric = np.diag(aleatoric)
            aleatorics.append(aleatoric)

        epistemic = np.vstack(epistemics)  # (batch_size, categories)
        aleatoric = np.vstack(aleatorics)  # (batch_size, categories)
        preds = torch.cat([i.unsqueeze(0) for i in preds]).cpu().detach().numpy()  # (batch_size, categories)

        return preds, aleatoric, epistemic
