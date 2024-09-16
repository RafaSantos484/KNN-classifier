import cv2
import numpy as np
from params import img_size


def get_img_features_arr(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img_hists = np.array([])
    num_channels = 1 if len(img.shape) == 2 else img.shape[2]
    for i in range(num_channels):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        img_hists = np.concatenate((img_hists, hist)).astype(int)
    return img_hists
