import pickle
from sklearn.neighbors import KNeighborsClassifier
from params import predict_imgs
from .utils import get_img_hists_arr


def run():
    knn_pickle = open('knn_pickle', 'rb')
    knn: KNeighborsClassifier = pickle.load(knn_pickle)
    knn_pickle.close()

    imgs_features = []
    for predict_img in predict_imgs:
        imgs_features.append(get_img_hists_arr(predict_img))

    predictions = knn.predict(imgs_features)
    print(predictions)
