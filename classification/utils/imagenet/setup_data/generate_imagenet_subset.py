import os
import numpy as np

import torchvision.datasets as datasets

data_dir = '/proj/vot2021/x_fahkh/joseph/imagenet'

# Data loading code
traindir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(traindir, None)
classes = train_dataset.classes
print("the number of total classes: {}".format(len(classes)))

seed = 1993
np.random.seed(seed)
subset_num = 100
subset_classes = np.random.choice(classes, subset_num, replace=False)
print("the number of subset classes: {}".format(len(subset_classes)))
print(subset_classes)

des_root_dir = '/proj/vot2021/x_fahkh/joseph/xLantern/data/seed_{}_subset_{}_imagenet/data/'.format(seed, subset_num)
if not os.path.exists(des_root_dir):
    os.makedirs(des_root_dir)
phase_list = ['train', 'val']
for phase in phase_list:
    if not os.path.exists(os.path.join(des_root_dir, phase)):
        os.mkdir(os.path.join(des_root_dir, phase))
    for sc in subset_classes:
        src_dir = os.path.join(data_dir, phase, sc)
        des_dir = os.path.join(des_root_dir, phase, sc)
        cmd = "cp -r {} {}".format(src_dir, des_dir)
        print(cmd)
        os.system(cmd)

print("Generation finished.")