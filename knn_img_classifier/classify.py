import pickle
from sklearn.neighbors import KNeighborsClassifier
from params import classify_imgs
from .utils import get_img_features_arr


def run():
    knn_pickle = open('knn_pickle', 'rb')
    knn: KNeighborsClassifier = pickle.load(knn_pickle)
    knn_pickle.close()

    imgs_features = []
    for classify_img in classify_imgs:
        imgs_features.append(get_img_features_arr(classify_img))

    classes = knn.predict(imgs_features)
    print(classes)
