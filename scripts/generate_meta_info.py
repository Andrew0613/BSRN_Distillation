from os import path as osp
from PIL import Image

from basicsr.utils import scandir


# generated name meta_info
def generate_meta_info_ntire2022stereo():

    gt_folder = 'datasets/NTIRE2022_StereoSR/Train/HR_sub'
    meta_info_txt = 'hct/data/meta_info/meta_info_ntire2022stereo_sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))
    key_list = []
    for img_n in img_list:
        n_idx = img_n.split('_')[0]
        s_idx = img_n.split('.')[0].split('_')[-1]
        key_list.append(f'{n_idx} {s_idx}')

    with open(meta_info_txt, 'w') as f:
        for idx, key in enumerate(key_list):
            print(idx + 1, key)
            f.write(f'{key}\n')


# generated path meta_info
def generate_meta_info_ntire2022stereo_valid():

    gt_folder = 'datasets/NTIRE2022_StereoSR/Validation/HR'
    meta_info_txt = 'hct/data/meta_info/meta_info_ntire2022stereo_Valid_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


# generated path meta_info
def generate_standard_meta_info_ntire2022stereo():

    gt_folder = 'datasets/NTIRE2022_StereoSR/Train/HR_sub'
    meta_info_txt = 'hct/data/meta_info/standard_meta_info_ntire2022stereo_sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

import os
# generated path meta_info
def generate_meta_info_LSDIR():

    gt_folder = '/mnt/petrelfs/puyuandong/Low_level_vision/dataset/LSDIR/Train/HR'
    meta_info_txt = '/mnt/petrelfs/puyuandong/Low_level_vision/dataset/LSDIR/Train/meta_info_LSDIR_GT.txt'
    img_list = sorted(list(scandir(gt_folder, recursive=True, full_path=False)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

if __name__ == '__main__':
    # generate_meta_info_ntire2022stereo()
    # generate_meta_info_ntire2022stereo_valid()
    # generate_standard_meta_info_ntire2022stereo()
    generate_meta_info_LSDIR()