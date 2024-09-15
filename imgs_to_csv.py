import os
import numpy as np
import pandas as pd
from PIL import Image
from params import folders, num_imgs, img_size


def images_to_csv(folder: str, num_imgs=-1, img_size=-1):
    print(f'Reading {folder} images...')
    img_files = os.listdir(folder)
    if num_imgs != -1:
        img_files = img_files[:num_imgs]

    imgs_data = []
    output_csv = f'{folder}.csv'
    processed_imgs = 0
    max_imgs_data_len = 100
    for img_file in img_files:
        img_path = os.path.join(folder, img_file)
        img = Image.open(img_path)
        if img_size != -1:
            img = img.resize((img_size, img_size))
        img_arr = np.array(img).flatten()
        imgs_data.append(img_arr)

        if len(imgs_data) == max_imgs_data_len:
            df = pd.DataFrame(imgs_data)
            df.to_csv(output_csv, index=False, mode='a')
            imgs_data = []
            processed_imgs += max_imgs_data_len
            print(f'progress: {processed_imgs / len(img_files) * 100:.2f}%')

    if len(imgs_data) != 0:
        df = pd.DataFrame(imgs_data)
        df.to_csv(output_csv, index=False, mode='a')


for folder in folders:
    if os.path.exists(f'{folder}.csv'):
        os.remove(f'{folder}.csv')

for folder in folders:
    images_to_csv(folder, num_imgs, img_size)
