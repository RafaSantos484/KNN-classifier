import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from params import classify_imgs_folder
from .utils import get_img_features_arr


def run():
    knn_pickle = open('knn_pickle', 'rb')
    knn: KNeighborsClassifier = pickle.load(knn_pickle)
    knn_pickle.close()

    imgs_features = []
    classify_imgs = os.listdir(classify_imgs_folder)
    for classify_img in classify_imgs:
        path = os.path.join(classify_imgs_folder, classify_img)
        imgs_features.append(get_img_features_arr(path))

    classes = knn.predict(imgs_features)
    for i in range(len(classify_imgs)):
        print(f'{classify_imgs[i]}: {classes[i]}')
