import pickle
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from params import img_size, predict_imgs


def run():
    knn_pickle = open('knn_pickle', 'rb')
    knn: KNeighborsClassifier = pickle.load(knn_pickle)
    knn_pickle.close()

    imgs_arrs = []
    for predict_img in predict_imgs:
        img = Image.open(predict_img).resize((img_size, img_size))
        img_arr = np.array(img).flatten()
        imgs_arrs.append(img_arr)

    predictions = knn.predict(imgs_arrs)
    print(predictions)
