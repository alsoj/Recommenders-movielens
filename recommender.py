import os

import tensorflow as tf
import pandas as pd
from tensorflow.python.client import device_lib

print("Tensorflow Version:", tf.VERSION)
devices = device_lib.list_local_devices()
print([x.name for x in devices])

num_cpus = os.cpu_count()
print("Num CPUs:", num_cpus)
