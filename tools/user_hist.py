import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    font_path = "/usr/share/fonts/wps-office/wps-office/times.ttf"
    font = FontProperties(fname=font_path)

    plt.style.use('seaborn-pastel')

    arrays = np.load('arrays.npy')
    n = arrays.shape[0]
    # random_indices = np.random.choice(n, 10000)
    # arrays = arrays[random_indices]

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_arrays = scaler.fit_transform(arrays)

    x = np.zeros_like(normalized_arrays)
    for i in range(arrays.shape[1]):
        pos_arrays = normalized_arrays[:, i]
        pos_arrays = pos_arrays[pos_arrays > 0]
        x[:, i][normalized_arrays[:, i] > 0], lam = stats.boxcox(pos_arrays)
        x[:, i][normalized_arrays[:, i] == 0] = -1 / lam

    x = scaler.fit_transform(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    labels = ['time_mean', 'time_std', 'overall_mean', 'overall_std', 'verified']
    for k, label in enumerate(labels):
        n, bins, patches = ax1.hist(normalized_arrays[:, k], alpha=0.7, label=label)
        for i in range(len(n)):
            ax1.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.03, int(n[i]), ha='center', va='bottom')

    ax1.set_title('Raw', fontproperties=font, fontsize=20)
    ax1.legend()

    labels = ['time_mean', 'time_std', 'overall_mean', 'overall_std', 'verified']
    for k, label in enumerate(labels):
        n, bins, patches = ax2.hist(x[:, k], alpha=0.7, label=label)
        for i in range(len(n)):
            ax2.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.03, int(n[i]), ha='center', va='bottom')

    ax2.set_title('After BOX-COX', fontproperties=font, fontsize=20)
    ax2.legend()

    plt.tight_layout()
    plt.show()
