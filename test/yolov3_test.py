import torch
from tqdm import tqdm
import os
import copy
import numpy as np
from data import Yolov3Data
from torch.utils.data import DataLoader
from common.utils import read_yaml, non_max_suppression, scale_coords, xyxy2xywh_np, xywh2xyxy, voc_ap
from pathlib import Path


def test(model, conf_thres=0.001, nms_thres=0.5):
    device = "cuda:0"
    classes = ['person']
    if model is None:
        pass
    else:
        device = model.device  # get model device
        data_parameters = read_yaml(model.data_yaml)
        classes = data_parameters['classes']
        batch_size = model.batch_size
    # Dataset
    data_parameters["data_path"] = '/home/lingc1/data/sports-training-data/player_detection/validate_dataset_5k_half_size'
    data_set = Yolov3Data(data_parameters, None, index_file='val_test')
    dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=0, collate_fn=data_set.collate_fn)
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p_80, r_80, f1_80, mp_80, mr_80, map_80, mf1_80 = 0., 0., 0., 0., 0., 0., 0., 0.
    seen = 0
    images_num = 0
    output_results = ""
    class_recs = {}
    nc = len(classes)
    names = classes
    det_lines = []
    imagenames = []
    npos_cls = {}
    for i in range(nc):
        npos_cls[i] = 0
        class_recs[i] = {}

    for i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_number = len(targets)

        preds = model.inference(imgs, None)

        output = non_max_suppression(preds, conf_thres=conf_thres, nms_thres=nms_thres)
        true_targets = targets[torch.sum(targets[:, 1:6], 1) != 0]

        # Statistics per image
        # remove the targets that fills 0 for data distribution.
        true_targets = targets[torch.sum(targets[:, 1:6], 1) != 0]
        # npos += len(true_targets)

        for si, pred in enumerate(output):
            images_num += 1
            if pred is not None and len(pred) > 0:
                # Rescale boxes from 416 to true image size
                # pred[:, :4] = scale_coords(imgs.shape[2:], pred[:, :4], im0_shape).round()
                for *xyxy, conf, cls_conf, cls in pred:
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int((xyxy[2] - xyxy[0]).round())
                    h = int((xyxy[3] - xyxy[1]).round())
                    output_line = "{:s},{:d},{:d},{:d},{:d},{:f}\n".format(
                        Path(str(si)).name, x, y, w,
                        h, conf)
                    det_lines.append(output_line)
                    output_results = output_results + output_line
            labels = true_targets[true_targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 4].tolist() if nl else []  # target class
            seen += 1

            imagename = os.path.splitext(Path(str(si)).name)[0]
            imagenames.append(imagename)
            if nl:

                unique_classes = np.unique(tcls).astype('int32')
                # detected = []
                for cls in unique_classes:
                    cls_idx = np.where(tcls == cls)[0]
                    tcls_tensor = labels[:, 4]
                    tcls_tensor = tcls_tensor[cls_idx]
                    npos_cls[cls] += len(tcls_tensor)
                    # target boxes
                    tbox = xywh2xyxy(labels[:, 0:4])
                    tbox = tbox[cls_idx]
                    tbox[:, [0, 2]] *= imgs.shape[3]
                    tbox[:, [1, 3]] *= imgs.shape[2]
                    bbox = np.array(tbox.cpu().numpy().round(), dtype=int)
                    det = [False] * len(tcls_tensor)
                    difficult = np.array(det)
                    class_recs[cls][imagename] = {'bbox': bbox,
                                                  'difficult': difficult,
                                                  'det': det}
    p_80, r_80, ap_80, f1_80 = [], [], [], []
    p_50, r_50, ap_50, f1_50 = [], [], [], []
    ap_80_iou = 0.8
    ap_50_iou = 0.5
    class_recs_80 = copy.deepcopy(class_recs)
    class_recs_50 = copy.deepcopy(class_recs)
    for cls in range(nc):
        rec_cls, prec_cls, ap_cls = voc_eval(det_lines, npos_cls[cls], imagenames, class_recs_80[cls],
                                             ovthresh=ap_80_iou, use_07_metric=True)
        f1_cls = 2 * prec_cls[-1] * rec_cls[-1] / (prec_cls[-1] + rec_cls[-1] + 1e-16)
        p_80.append(prec_cls[-1])
        r_80.append(rec_cls[-1])
        ap_80.append(ap_cls)
        f1_80.append(f1_cls)
        print("AP 80")
        print("person ap is: %.6f" % (ap_cls * 100))
        print("recall is %.6f" % (rec_cls[-1] * 100))
        print("precision is %.6f" % (prec_cls[-1] * 100))
        rec_cls, prec_cls, ap_cls = voc_eval(det_lines, npos_cls[cls], imagenames, class_recs_50[cls],
                                             ovthresh=ap_50_iou, use_07_metric=True)
        f1_cls = 2 * prec_cls[-1] * rec_cls[-1] / (prec_cls[-1] + rec_cls[-1] + 1e-16)
        p_50.append(prec_cls[-1])
        r_50.append(rec_cls[-1])
        ap_50.append(ap_cls)
        f1_50.append(f1_cls)
        print("AP 50")
        print("person ap is: %.6f" % (ap_cls * 100))
        print("recall is %.6f" % (rec_cls[-1] * 100))
        print("precision is %.6f" % (prec_cls[-1] * 100))
    mp_80, mr_80, map_80, mf1_80 = np.mean(p_80) * 100, np.mean(r_80) * 100, np.mean(ap_80) * 100, np.mean(f1_80) * 100
    mp_50, mr_50, map_50, mf1_50 = np.mean(p_50) * 100, np.mean(r_50) * 100, np.mean(ap_50) * 100, np.mean(f1_50) * 100
    # Print results
    all_target_sum = 0
    for _, cls_npos in npos_cls.items():
        all_target_sum += cls_npos
    pf = '%20s' + '%10.6g' * 6  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')
    print(pf % ('all', seen, all_target_sum, mp_80, mr_80, map_80, mf1_80), end='\n\n')
    print(pf % ('all', seen, all_target_sum, mp_50, mr_50, map_50, mf1_50), end='\n\n')

    # Print results per class
    # if nc > 1:
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, npos_cls[c], p[i], r[i], ap[i], f1[i]))
    if nc > 1:
        for i in range(nc):
            print(pf % (names[i], seen, npos_cls[i], p_80[i], r_80[i], ap_80[i], f1_80[i]))
            print(pf % (names[i], seen, npos_cls[i], p_50[i], r_50[i], ap_50[i], f1_50[i]))
    # Return results
    maps = np.zeros(nc)
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    for i in range(nc):
        maps[i] = ap_80[i]
    return (mp_80, mr_80, map_80, mf1_80, loss / len(dataloader), mp_50, mr_50, map_50, mf1_50), maps


def voc_eval(lines, npos, imagenames, class_recs, ovthresh=0.5, use_07_metric=False, is_half_size=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    splitlines = [x.strip().split(',') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[5]) for x in splitlines])
    if is_half_size:
        BB = np.array([[float(z) * 2 for z in x[1:5]] for x in splitlines])
    else:
        BB = np.array([[float(z) for z in x[1:5]] for x in splitlines])
    # if len(BB) == 0:
    #     return 0, 0, 0
    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    if len(BB) > 0:
        BB = BB[sorted_ind, :]
    # else:

    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        fn, ex = os.path.splitext(image_ids[d])
        if is_half_size:
            fn = fn[fn.find("hs_") + 3:]
        if fn not in imagenames:
            continue
        R = class_recs[fn]
        if len(BB) > 0:
            bb = BB[d, :].astype(float)
        else:
            bb = [0, 0, 0, 0]
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2] + bb[0] - 1)
            iymax = np.minimum(BBGT[:, 3], bb[3] + bb[1] - 1)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] + 1.) * (bb[3] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
                    # print("")
                    # print(bb)
                    # print(ovmax)
        else:
            fp[d] = 1.
            # print("")
            # print(bb)
            # print(ovmax)

    # compute precision recall
    # print("cvt_tp:")
    # print(len(tp))
    # print(tp)
    # print("cvt_fp:")
    # print(len(fp))
    # print(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if len(prec) == 0:
        prec = np.array([0])
    if len(rec) == 0:
        rec = np.array([0])
    return rec, prec, ap
