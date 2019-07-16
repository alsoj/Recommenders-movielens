import os

import tensorflow as tf
import pandas as pd
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

data = pd.read_csv('./data/10M/ratings.dat', sep="::", header=None, engine='python')

print(data[:10])

# data = load_pandas_df(
#     size=MOVIELENS_DATA_SIZE,
#     header=[USER_COL, ITEM_COL, RATING_COL],
#     genres_col='Genres_string'  # load genres as a temporal column 'Genres_string'
# )
data.head()