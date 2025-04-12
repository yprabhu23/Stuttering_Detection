#This file is used to collect information on certain things in the model.
#Ex. number of samples in a dataset.


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models

DATASET_PATH = '/home/yashprabhu/Documents/SortedByProlongation'
data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
train_files = filenames
val_files = filenames
test_files = filenames
print('Number of total examples:', num_samples)

print('Example file tensor:', filenames[0])