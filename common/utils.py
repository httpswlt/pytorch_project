# coding:utf-8
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import torch


def read_yaml(path):
    """

    :param path: the path of yaml file
    :return:
    """
    if not os.path.exists(path):
        logging.error('{} is not exist. '.format(path))
        exit(0)

    with open(path, 'r', encoding='utf-8') as f:
        net = yaml.load(f)

    if 'anchors' in net.keys():
        temp = []
        for layer_anchors in net['anchors']:
            temp.append([eval(anchor) for anchor in layer_anchors])
        net['anchors'] = temp

    return net


def parse_xml(xml_path, classes):
    """

    :param xml_path:    the path of xml
    :param classes:     all classes, e.g. ['a', 'b', ...]
    :return:    [[x, y, w, h, cls_id], ...]
    """
    tree = ET.parse(open(xml_path, 'rb'))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    result = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = xyxy2xywh(b)
        result.append(list(bb) + [cls_id])
    result = np.array(result)
    return result


def xyxy2xywh(box):
    """

    :param box: (x1, y1, x2, y2), type:tuple
    :return:    (center_x, center_y, w, h)
    """
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h


def normalization_coordinate(targets, img_shape):
    """

    :param targets:  e.g.: [[cen_x, cen_y, w, h, cls],....], type: numpy.ndarray
    :param img_shape  (h, w, c)
    :return:
    """
    img_h, img_w, img_c = img_shape
    dx = 1. / img_w
    dy = 1. / img_h
    targets[:, 0] *= dx
    targets[:, 1] *= dy
    targets[:, 2] *= dx
    targets[:, 3] *= dy
    return targets


def scale_coords(src_shape, coords, dst_shape):
    """
        # Rescale coords1 (xyxy) from src_shape to dst_shape
    :param dst_shape: scr image's shape
    :param coords:  the coordinate that you want to convert.
    :param src_shape:  dst image's shape
    :return:
    """

    gain = max(src_shape) / max(dst_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (src_shape[1] - dst_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (src_shape[0] - dst_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def constrain_coordinate(min_v, max_v, coord):
    """

    :param min_v: min value, e.g.: 0
    :param max_v: max value, e.g.: 1
    :param coord:   e.g.: [[cen_x, cen_y, w, h, cls],....], type: numpy.ndarray
    :return:
    """
    cen_x = coord[:, 0]
    cen_y = coord[:, 1]
    w = coord[:, 2]
    h = coord[:, 3]
    left = cen_x - w / 2
    right = cen_x + w / 2
    top = cen_y - h / 2
    bottom = cen_y + h / 2
    constrains = [left, right, top, bottom]
    for i, point in enumerate(constrains):
        index = np.where(point < min_v)
        constrains[i][index] = min_v
        index = np.where(point > max_v)
        constrains[i][index] = max_v

    w = right - left
    h = bottom - top
    constrains = [w, h]
    for i, point in enumerate(constrains):
        index = np.where(point < min_v)
        constrains[i][index] = min_v
        index = np.where(point > max_v)
        constrains[i][index] = max_v

    cen_x = (left + right) / 2.
    cen_y = (top + bottom) / 2.
    coord[:, :-1] = np.vstack((cen_x, cen_y, w, h)).T
    # constrain w and h
    remove_index = np.where(coord[:, 2] < 0.001)[0]
    coord = np.delete(coord, remove_index, axis=0)
    remove_index = np.where(coord[:, 3] < 0.001)[0]
    coord = np.delete(coord, remove_index, axis=0)
    return coord


def xywh2xyxy(x):
    # Convert bounding box format from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh_np(x):
    # Convert bounding box format from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] + x[:, 1]
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    inter_area = (torch.min(b1_x2.float(), b2_x2.float()) - torch.max(b1_x1.float(), b2_x1.float())).clamp(0) * \
                 (torch.min(b1_y2.float(), b2_y2.float()) - torch.max(b1_y1.float(), b2_y1.float())).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area.float()

    return inter_area / union_area


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]
        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]
        det_max = []
        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold
            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
            # output[image_i] = pred
    return output


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    path = '../config/net.yaml'
    temp = read_yaml(path)
    print(temp)
