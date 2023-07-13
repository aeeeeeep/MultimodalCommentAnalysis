import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    font_path = "/usr/share/fonts/wps-office/wps-office/times.ttf"
    font = FontProperties(fname=font_path)

    plt.style.use('seaborn-pastel')

    arrays = np.load('arrays.npy')
    n = arrays.shape[0]
    # random_indices = np.random.choice(n, 10000)
    # arrays = arrays[random_indices]
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

    silhouette_avg = silhouette_score(x_pca, labels)
    print("Silhouette Score:", silhouette_avg)

    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='Set3')

    plt.title('K-means clustering', fontproperties=font, fontsize=20)
    plt.text(x=-1, y=0, s='Non-malicious Comments', fontproperties=font, fontdict={'weight': 'normal', 'size': 20})
    plt.text(x=1.5, y=2.5, s='malicious Comments', fontproperties=font, fontdict={'weight': 'normal', 'size': 20})

    plt.show()
