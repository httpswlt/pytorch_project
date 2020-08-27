# coding:utf-8
import torch
from torch import nn

from loss.utils import bbox_iou, bbox_wh_iou


class YOLOLoss(nn.Module):
    """
        The header of YOLO
    """

    def __init__(self, num_classes, image_size, ignore_thres_first_loss):
        """

        :param num_classes:
        :param image_size:
        :param ignore_thres_first_loss:
        """
        super(YOLOLoss, self).__init__()

        self.batch_size = 0
        self.grid_h = 0
        self.grid_w = 0
        self.scale_anchors = []
        self.stride_h = 0
        self.stride_w = 0

        self.prediction = None
        self.pred_center_x = 0
        self.pred_center_y = 0
        self.pred_w = 0
        self.pred_h = 0
        self.pred_conf = 0
        self.pred_cls = 0

        self.grid_x = None
        self.grid_y = None
        self.anchor_w = None
        self.anchor_h = None
        self.device = None

        self.num_classes = num_classes
        self.img_width = int(image_size[0])
        self.img_height = int(image_size[1])
        self.ignore_threshold = 0.5
        self.ignore_thres_first_loss = ignore_thres_first_loss  # ignore threshold for first loss
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.obj_scale = 1
        self.noobj_scale = 0.5

        self.anchors = None
        self.num_anchors = None

    def set_anchors(self, anchors):
        """

        :param anchors:
        :return:
        """
        self.anchors = anchors
        self.num_anchors = len(anchors)

    def _configure_grid_anchors(self):
        self.grid_x = torch.linspace(0, self.grid_w - 1, self.grid_w).repeat((self.grid_h, 1)).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.pred_center_x.shape).to(self.device)
        self.grid_y = torch.linspace(0, self.grid_h - 1, self.grid_h).repeat((self.grid_w, 1)).permute(1, 0).repeat(
            (self.batch_size * self.num_anchors, 1, 1)).view(self.pred_center_y.shape).to(self.device)
        self.anchor_w = torch.tensor(self.scale_anchors).to(
            self.device).index_select(1, torch.tensor([0]).to(self.device))
        self.anchor_h = torch.tensor(self.scale_anchors).to(
            self.device).index_select(1, torch.tensor([1]).to(self.device))
        self.anchor_w = self.anchor_w.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(
            self.pred_w.shape)
        self.anchor_h = self.anchor_h.repeat(self.batch_size, 1).repeat(1, 1, self.grid_h * self.grid_w).view(
            self.pred_h.shape)

    def forward(self, inputs):
        """

        :param inputs: type: tuple, (x, target)
        :return:
        """
        x = inputs[0]
        self.device = x.device
        if inputs[1] is None:
            target = None
        else:
            target = inputs[1].clone()
            target = target[torch.sum(target[:, 1:6], 1) != 0]
        self.batch_size = x.size(0)
        self.grid_h = x.size(2)
        self.grid_w = x.size(3)
        self.stride_h = self.img_height / self.grid_h
        self.stride_w = self.img_width / self.grid_w

        self.scale_anchors = [(w / self.stride_w, h / self.stride_h) for w, h in self.anchors]

        self.prediction = (
            x.view(self.batch_size, self.num_anchors, self.num_classes + 5, self.grid_h, self.grid_w)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # get outputs
        self.pred_center_x = torch.sigmoid(self.prediction[..., 0])  # center x
        self.pred_center_y = torch.sigmoid(self.prediction[..., 1])  # center y
        self.pred_w = self.prediction[..., 2]  # predict box width
        self.pred_h = self.prediction[..., 3]  # predict box height
        self.pred_conf = torch.sigmoid(self.prediction[..., 4])  # box confidence
        self.pred_cls = torch.sigmoid(self.prediction[..., 5:])  # object confidence of per classify
        self._configure_grid_anchors()

        output = self.__detect()
        if target is None or len(target) == 0:
            return output, 0, 0, 0, 0, 0

        # get ignore_mask of first loss in darknet
        ignore_mask = self._first_loss(self.prediction, target)
        # build target
        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, scale = self.__build_target(target)
        # merge ignore_mask and obj_mask
        ignore_mask[obj_mask == 1] = 1
        # merge ignore_mask and noobj_mask
        noobj_mask[ignore_mask == 0] = 0
        loss_conf_obj = 0
        loss_conf_noobj = 0
        loss_x = 0
        loss_y = 0
        loss_w = 0
        loss_h = 0
        loss_cls = 0
        if torch.sum(obj_mask) > 0:
            # Loss:
            # x, y, w, h losses
            loss_x = torch.unsqueeze(
                torch.sum(
                    self.mse_loss(self.pred_center_x[obj_mask], tx[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
            loss_y = torch.unsqueeze(
                torch.sum(
                    self.mse_loss(self.pred_center_y[obj_mask], ty[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
            loss_w = torch.unsqueeze(
                torch.sum(self.mse_loss(self.pred_w[obj_mask], tw[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
            loss_h = torch.unsqueeze(
                torch.sum(self.mse_loss(self.pred_h[obj_mask], th[obj_mask]) * scale[obj_mask] * scale[obj_mask]), 0)
            # confidence losses
            loss_conf_obj = torch.unsqueeze(torch.sum(self.mse_loss(self.pred_conf[obj_mask], tconf[obj_mask])), 0)
            # classify losses
            loss_cls = torch.unsqueeze(torch.sum(self.mse_loss(self.pred_cls[obj_mask], tcls[obj_mask])), 0)

        if torch.sum(noobj_mask) > 0:
            loss_conf_noobj = torch.unsqueeze(torch.sum(self.mse_loss(self.pred_conf[noobj_mask], tconf[noobj_mask])),
                                              0)

        loss_conf = loss_conf_obj + loss_conf_noobj

        # total losses
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        loss_xy = loss_x + loss_y
        loss_wh = loss_w + loss_h

        return output, total_loss, loss_xy, loss_wh, loss_conf, loss_cls

    def __build_target(self, target):
        """

        :param target: type: list: [[image_num, x, y, w, h, label],...]
        :return:
        """
        obj_mask = torch.empty((self.batch_size, self.num_anchors, self.grid_h, self.grid_w),
                               dtype=torch.bool).fill_(0).to(self.device)
        noobj_mask = torch.empty((self.batch_size, self.num_anchors, self.grid_h, self.grid_w),
                                 dtype=torch.bool).fill_(1).to(self.device)
        tx = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0).to(self.device)
        ty = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0).to(self.device)
        tw = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0).to(self.device)
        th = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0).to(self.device)
        tconf = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(0).to(self.device)
        tcls = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w, self.num_classes).fill_(
            0).to(self.device)
        scale = torch.empty(self.batch_size, self.num_anchors, self.grid_h, self.grid_w).fill_(1).to(self.device)

        # map target coordinate to grid size
        target_bbox = target[..., ::] * 1
        gwh_iou = target_bbox[..., 3:-1].clone()  # For best iou choose.
        target_bbox[..., 1:-1:2] *= self.grid_w
        target_bbox[..., 2:-1:2] *= self.grid_h
        gxy = target_bbox[:, 1:3]
        gwh = target_bbox[..., 3:-1]
        img_num_ = target_bbox[:, 0].long()
        img_num_ = img_num_ - img_num_[0]
        target_label = target_bbox[:, -1].long()

        # compute iou
        ious = torch.stack(tuple([bbox_wh_iou(anchor, gwh_iou) for anchor in
                                  torch.div(torch.tensor(self.anchors).to(self.device),
                                            torch.tensor((self.img_width,
                                                          self.img_height), dtype=torch.float32).to(self.device))]), 0)
        best_ious, best_index = ious.max(0)
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi_, gj_ = gxy.long().t()
        # Init scale weight 2- truth.x * truth.y
        gw_iou, gh_iou = gwh_iou.t()
        sc = 2 - gw_iou * gh_iou

        outlier_filter_std = (gi_ < self.grid_w).long() + (gj_ < self.grid_h).long() + (gi_ >= 0).long() + (
                gj_ >= 0).long()
        best_index_filter = (outlier_filter_std == 4)

        img_num, best_index, gi, gj = img_num_[best_index_filter], best_index[best_index_filter], gi_[
            best_index_filter], gj_[best_index_filter]
        gx, gy = gx[best_index_filter], gy[best_index_filter]
        gw, gh = gw[best_index_filter], gh[best_index_filter]
        target_label = target_label[best_index_filter]
        sc = sc[best_index_filter]

        if len(best_index):
            # Set masks
            obj_mask[img_num, best_index, gj, gi] = 1
            noobj_mask[img_num, best_index, gj, gi] = 0

            # Coordinates
            tx[img_num, best_index, gj, gi] = gx - gx.floor()
            ty[img_num, best_index, gj, gi] = gy - gy.floor()

            # Width and height
            tw[img_num, best_index, gj, gi] = torch.log(
                gw * self.stride_w / torch.tensor(self.anchors).to(self.device)[best_index][:, 0] + 1e-16)
            th[img_num, best_index, gj, gi] = torch.log(
                gh * self.stride_h / torch.tensor(self.anchors).to(self.device)[best_index][:, 1] + 1e-16)
            # One-hot encoding of label
            tcls[img_num, best_index, gj, gi, target_label] = 1

            # object
            tconf[img_num, best_index, gj, gi] = 1

            # Scale weight
            scale[img_num, best_index, gj, gi] = sc

        # Set noobj mask to zero where iou exceeds ignore threshold(paper said, but darknet doesn't have)
        # for i, anchor_ious in enumerate(ious[self.mask, :].t()):
        #    noobj_mask[img_num_[i], anchor_ious > self.ignore_threshold, gj_[i], gi_[i]] = 0

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, scale

    def _first_loss(self, pred, target):
        """

        :param pred: type: tensor: tensor.size([image_num, anchor_num, grid_j, gird_i, 5+class_num])
        :param target: type: list: [[image_num, x, y, w, h, cls],...]
        :return: ignore_mask which ignores iou(pred, truth)  > ignore_thres_first_loss
        """

        # Init ignore_mask which ignores iou(pred, truth)  > ignore_thres_first_loss
        ignore_mask = torch.empty((self.batch_size, self.num_anchors, self.grid_h, self.grid_w),
                                  dtype=torch.bool).fill_(1).to(
            self.device)

        p_boxes = torch.zeros_like(pred[0])
        index_start = target[0][0]
        for i, pi0 in enumerate(pred):
            p_boxes = p_boxes.view(pi0.size()[0], pi0.size()[1], pi0.size()[2], 5 + self.num_classes)
            t = target[target[..., 0] == (i + index_start)]  # Targets for image j of batchA
            if len(t):
                # transform pred to yolo box
                p_boxes[..., 0] = (torch.sigmoid(pi0[..., 0]) + self.grid_x[i]) / self.grid_w
                p_boxes[..., 1] = (torch.sigmoid(pi0[..., 1]) + self.grid_y[i]) / self.grid_h
                p_boxes[..., 2] = (torch.exp(pi0[..., 2]) * self.anchor_w[i]) / self.grid_w
                p_boxes[..., 3] = (torch.exp(pi0[..., 3]) * self.anchor_h[i]) / self.grid_h
                p_boxes = p_boxes.view(pi0.size()[0] * pi0.size()[1] * pi0.size()[2], 5 + self.num_classes)

                # compute iou for each pred gird and all targets.
                ious = torch.stack(tuple([bbox_iou(x, p_boxes[:, :4], False) for x in t[:, 1:5]]))
                best_ious, best_index = ious.max(0)
                best_ious, best_index = best_ious.view(pi0.size()[0], pi0.size()[1], pi0.size()[2],
                                                       1), best_index.view(pi0.size()[0], pi0.size()[1],
                                                                           pi0.size()[2], 1)
                ignore_mask[i][torch.squeeze(best_ious > self.ignore_thres_first_loss, 3)] = 0
                del ious

        return ignore_mask

    def __detect(self):

        # add offset
        pred_boxes = torch.empty(self.prediction[..., :4].shape).to(self.device)
        pred_boxes[..., 0] = self.pred_center_x + self.grid_x
        pred_boxes[..., 1] = self.pred_center_y + self.grid_y
        pred_boxes[..., 2] = torch.exp(self.pred_w) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(self.pred_h) * self.anchor_h

        _scale = torch.tensor([self.stride_w, self.stride_h] * 2).to(self.device)
        output = torch.cat(
            (
                pred_boxes.view((self.batch_size, -1, 4)) * _scale,
                self.pred_conf.view((self.batch_size, -1, 1)),
                self.pred_cls.view((self.batch_size, -1, self.num_classes)),
            ),
            -1,
        )
        return output


if __name__ == '__main__':
    from backbone.darknet import darknet53
    from header import Yolov3Header

    mask1 = [7, 8, 6]
    anchors1 = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors1 = anchors1.strip().split(",")
    anchors1 = [float(a) for a in anchors1]
    ANCHORS1 = [(anchors1[i], anchors1[i + 1]) for i in range(0, len(anchors1), 2)]
    anchors1 = [ANCHORS1[i] for i in mask1]
    classes = 1
    inputss = torch.ones((1, 3, 256, 256))
    # 1: construct backbone network
    backbone = darknet53()
    # 2: construct header
    in_channels = backbone.get_last_output_channels()
    out_channels = len(mask1) * (5 + classes)
    yolo_header = Yolov3Header(in_channels, out_channels, backbone)
    # 3: construct loss
    inference = yolo_header(inputss)
    ignore_thresh = .7
    yolo = YOLOLoss(classes, (256, 256), ignore_thresh)
    # 4: regression the coordination according to the loss function.
    yolo.set_anchors(anchors1)
    yolo((inference[0], torch.tensor([[0, 0.0117, 0.0156, 0.0195, 0.0234, 0],
                                      [0, 0.0017, 0.0156, 0.0095, 0.0234, 0]])))
