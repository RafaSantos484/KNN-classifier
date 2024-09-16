import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from params import folders, test_size, random_state, max_k


def run():
    df = pd.DataFrame()
    classes = np.array([])
    for folder in folders:
        folder_df = pd.read_csv(f'{folder}.csv')
        df = pd.concat([df, folder_df])
        classes = np.concatenate(
            [classes, np.full(folder_df.shape[0], folder)])

    x_train, x_test, y_train, y_test = train_test_split(
        df.values, classes, test_size=test_size, random_state=random_state)

    best_knn = None
    best_score = -1
    for i in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        score = knn.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_knn = knn
            print(f'new best score: {best_score} with {i} neighbors')

    knn_pickle = open('knn_pickle', 'wb')
    pickle.dump(best_knn, knn_pickle)
    knn_pickle.close()
    print('Exported knn model to knn_pickle')
