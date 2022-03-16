import os

import torchvision.datasets as datasets
from PIL import Image

src_root_dir = 'data/imagenet/data/'
des_root_dir = 'data/imagenet_resized_256/data/'
if not os.path.exists(des_root_dir):
    os.makedirs(des_root_dir)

phase_list = ['train', 'val']
for phase in phase_list:
    if not os.path.exists(os.path.join(des_root_dir, phase)):
        os.mkdir(os.path.join(des_root_dir, phase))
    data_dir = os.path.join(src_root_dir, phase)
    tg_dataset = datasets.ImageFolder(data_dir)
    for cls_name in tg_dataset.classes:
        if not os.path.exists(os.path.join(des_root_dir, phase, cls_name)):
            os.mkdir(os.path.join(des_root_dir, phase, cls_name))
    cnt = 0
    for item in tg_dataset.imgs:
        img_path = item[0]
        img = Image.open(img_path)
        img = img.convert('RGB')
        save_path = img_path.replace('imagenet', 'imagenet_resized_256')
        resized_img = img.resize((256,256), Image.BILINEAR)
        resized_img.save(save_path)
        cnt = cnt+1
        if cnt % 1000 == 0:
            print(cnt, save_path)

print("Generation finished.")