import os

import tensorflow as tf
import pandas as pd
import sklearn.preprocessing

from tensorflow.python.client import device_lib

print("Tensorflow Version:", tf.VERSION)
devices = device_lib.list_local_devices()
print([x.name for x in devices])

num_cpus = os.cpu_count()
print("Num CPUs:", num_cpus)

# Data column names
USER_COL = 'UserId'
ITEM_COL = 'MovieId'
RATING_COL = 'Rating'
ITEM_FEAT_COL = 'Genres'

#### Hyperparameters
MODEL_TYPE = 'wide_deep'
EPOCHS = 50  # if 0, only 1 batch will be processed
BATCH_SIZE = 64


### 1. Prepare Data
#### 1.1 Movie Rating and Genres Data
df_rating = pd.read_csv('./data/10M/ratings.dat', sep="::", header=None, names=[USER_COL, ITEM_COL, RATING_COL, 'timestamp'], engine='python')
df_movie = pd.read_csv('./data/10M/movies.dat', sep="::", header=None, names=[ITEM_COL, 'MovieName', 'Genres_string'], engine='python')

print('df_ratings \n', df_rating[:10])
print('df_movie \n', df_movie[:10])

df_data = pd.merge(df_rating, df_movie)

print('df_data \n', df_data[:10])


#### 1.2 Encode Item Features (Genres)
# Encode 'genres' into int array (multi-hot representation) to use as item features
genres_encoder = sklearn.preprocessing.MultiLabelBinarizer()
df_data[ITEM_FEAT_COL] = genres_encoder.fit_transform(
    df_data['Genres_string'].apply(lambda s: s.split("|"))
).tolist()
print("Genres:", genres_encoder.classes_)
print(df_data.drop_duplicates(ITEM_COL)[[ITEM_COL, 'Genres_string', ITEM_FEAT_COL]].head())