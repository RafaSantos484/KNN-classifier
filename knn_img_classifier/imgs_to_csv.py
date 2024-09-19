import os
import shutil
import pandas as pd
from params import folders, num_imgs, img_size
from .utils import get_img_features_arr


def images_to_csv(folder: str, num_imgs=-1, img_size=-1):
    print(f'Reading {folder} images...')
    img_files = os.listdir(folder)
    if num_imgs != -1:
        img_files = img_files[:num_imgs]

    imgs_features = []
    output_csv = f'{folder}.csv'
    processed_imgs = 0
    max_imgs_data_len = 100
    for img_file in img_files:
        img_path = os.path.join(folder, img_file)
        imgs_features.append(get_img_features_arr(img_path))
        if len(imgs_features) == max_imgs_data_len:
            df = pd.DataFrame(imgs_features)
            df.to_csv(f'csvs/{output_csv}', index=False, mode='a')
            imgs_features = []
            processed_imgs += max_imgs_data_len
            print(f'progress: {processed_imgs / len(img_files) * 100:.2f}%')

    if len(imgs_features) != 0:
        df = pd.DataFrame(imgs_features)
        df.to_csv(f'csvs/{output_csv}', index=False, mode='a')


def run():
    if os.path.exists('csvs'):
        shutil.rmtree('csvs')
    os.mkdir('csvs')

    for folder in folders:
        images_to_csv(folder, num_imgs, img_size)
