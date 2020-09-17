# coding: utf-8
import os
import random

data_path = '/home/lintaowx/data/st_middle/120'
val_num = 10000
train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'val')
os.makedirs(train_path, exist_ok=True)
for classify in os.listdir(data_path):
    if not classify.isdigit():
        continue
    classify_path = os.path.join(data_path, classify)
    classify_image = os.listdir(classify_path)
    random.shuffle(classify_image)
    val_image = random.sample(classify_image, val_num)
    
    new_path = os.path.join(val_path, classify)
    os.makedirs(new_path, exist_ok=True)
    for image in val_image:
        new_image_path = os.path.join(new_path, image)
        old_image_path = os.path.join(classify_path, image)
        command = 'mv {} {}'.format(old_image_path, new_image_path)
        # print(command)
        os.system(command)
