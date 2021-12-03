import pandas as pd
from datetime import datetime
import numpy as np


def data_to_csv(datafile, title, sep):
    df = pd.read_csv(datafile, sep=sep, header=None, engine='python')
    df.columns = title
    # print(df.head())
    # print(df.dtypes)
    # print(df.isnull().sum())
    if title[-1] == 'timestamp':
        ans = []
        for item in df['timestamp'].to_numpy():
            ans.append(str(datetime.fromtimestamp(item)).split(' ')[0])
        df['timestamp'] = pd.DataFrame(ans, columns=['timestamp'])
    savename = datafile.split('.')[1]
    df.to_csv(savename + '.csv', encoding='utf-8', index=False)
    # print('=====================================================================')


if __name__ == "__main__":
    files = {'u.data': [['user_id', 'item_id', 'rating', 'timestamp'], '\\t'],
             'u.item': [['item_id', 'movie_title', 'release_date', 'video_release_date',
                         'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western'], '|'],
             'u.user': [['user_id', 'age', 'gender', 'occupation', 'zip_code'], '|']
             }

    for k, v in files.items():
        data_to_csv(k, v[0], v[1])

    df_data = pd.read_csv('data.csv')
    df_item = pd.read_csv('item.csv')
    df_user = pd.read_csv('user.csv')
    df = pd.merge(df_data, df_user, on='user_id')
    df = pd.merge(df, df_item, on='item_id')
    df = df.drop(['user_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'zip_code'], axis=1)
    # print(df.isnull().sum())
    df['gender'] = pd.factorize(df['gender'])[0]
    df['occupation'] = pd.factorize(df['occupation'])[0] + 1
    df['rating'] = np.select([(df['rating'] <= 3), (df['rating'] >= 4)], [0, 1])
    df.to_csv('MovieLens_merged.csv', encoding='utf-8', index=False)
