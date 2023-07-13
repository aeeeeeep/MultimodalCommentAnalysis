import os

import numpy as np
import pandas as pd
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
    ]).reshape(-1, 5)


if __name__ == '__main__':
    data_directory = 'users'

    arrays = []
    for filename in tqdm(os.listdir(data_directory)):
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath):
            array = read_multi_line_json(filepath)
            arrays.append(array)

    arrays = np.vstack(arrays)
    np.save('arrays.npy', arrays)
