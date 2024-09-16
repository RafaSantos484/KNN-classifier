import cv2
import numpy as np
from params import img_size

def get_img_hists_arr(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img_hists = np.array([])
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        img_hists = np.concatenate((img_hists, hist)).astype(int)
    return img_hists
