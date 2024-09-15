import math
import pickle
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

knn_pickle = open('knn_pickle', 'rb')
knn: KNeighborsClassifier = pickle.load(knn_pickle)
knn_pickle.close()

img_size = int(math.sqrt(knn.n_features_in_ / 3))
gorilla_img = Image.open('gorilla.jpg').resize((img_size, img_size))
orangutan_img = Image.open('orangutan.jpg').resize((img_size, img_size))

gorilla_img_arr = np.array(gorilla_img).flatten()
orangutan_img_arr = np.array(orangutan_img).flatten()

predictions = knn.predict([gorilla_img_arr, orangutan_img_arr])
print(predictions)
