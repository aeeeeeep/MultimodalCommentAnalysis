import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def read_multi_line_json(file_path):
    with open(file_path, 'r') as file:
        json_list = ''.join(file.readlines())
    df = pd.read_json(json_list, lines=True)
    df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')
    df['reviewTime'] = df['reviewTime'].map(pd.Timestamp.timestamp)
    df = df.sort_values('reviewTime')
    df.fillna(0, inplace=True)
    try:
        verified = df['verified'].value_counts(normalize=True)[False] / len(df['verified'])
    except:
        verified = 0

    return np.array([
        df['reviewTime'].diff()[1:].mean(),
        df['reviewTime'].diff()[1:].std(),
        df['overall'].mean(),
        df['overall'].std(),
        verified,
    ]).reshape(-1, 5), df['reviewerID'][0]


if __name__ == '__main__':
    data_directory = 'users'

    arrays = []
    ids = []
    for filename in tqdm(os.listdir(data_directory)[:10000]):
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath):
            array, ID = read_multi_line_json(filepath)
            arrays.append(array)
            ids.append(ID)

    arrays = np.vstack(arrays)
    arrays = arrays[:10000]

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_arrays = scaler.fit_transform(arrays)

    x = np.zeros_like(normalized_arrays)
    for i in range(arrays.shape[1]):
        pos_arrays = normalized_arrays[:, i]
        pos_arrays = pos_arrays[pos_arrays > 0]
        x[:, i][normalized_arrays[:, i] > 0], lam = stats.boxcox(pos_arrays)
        x[:, i][normalized_arrays[:, i] == 0] = -1 / lam

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(x_pca)
    print("1: ", len(labels[labels == 1]))
    print("0: ", len(labels[labels == 0]))
    labels = np.asarray(labels, dtype=bool)
    ids = np.squeeze(ids)
    print("1: ", ids[labels][:5])
    print("0: ", ids[~labels][:5])
