import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(in_file, out_file):
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        if (np.array(bb) > 0).all():
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def convert_labeme_2_yolov5():
    img_root_path = '/ifs/data/Dataset/player/player_detection/training/JPEGImages'
    anno_root_path = '/ifs/data/Dataset/player/player_detection/training/Annotations'
    data_type = 'train'
    img_format = '.jpg'
    anno_format = '.xml'
    imgs = [name.split(img_format)[0] for name in os.listdir(img_root_path)]
    annotations = [name.split(anno_format)[0] for name in os.listdir(anno_root_path)]
    same_data = set(annotations).intersection(imgs)
    new_anno_dir = os.path.join('../data/person/labels', data_type)
    new_img_dir = os.path.join('../data/person/images', data_type)
    os.makedirs(new_anno_dir, exist_ok=True)
    os.makedirs(new_img_dir, exist_ok=True)
    
    for name in tqdm(same_data):
        if name != 'E16_10_29_20_48_35__LIVE_01C022F0378_1_3':
            continue
        img_path = os.path.join(img_root_path, '{}{}'.format(name, img_format))
        anno_path = os.path.join(anno_root_path, '{}{}'.format(name, anno_format))
        
        new_img_path = os.path.join(new_img_dir, '{}{}'.format(name, img_format))
        new_anno_path = os.path.join(new_anno_dir, '{}{}'.format(name, '.txt'))
        try:
            with open(new_anno_path, 'w') as f:
                convert_annotation(anno_path, f)
            command = 'cp -rf {} {}'.format(img_path, new_img_path)
            os.system(command)
        except Exception as e:
            command = 'rm -rf {} {}'.format(new_img_path, new_anno_path)
            os.system(command)


def main():
    convert_labeme_2_yolov5()


if __name__ == '__main__':
    classes = ['person']
    main()
