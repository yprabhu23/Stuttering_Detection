import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

FOLDER_PATH = '/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/[CLIP_DIR]/clips/HeStutters/0/'

DATASET_PATH = FOLDER_PATH

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin=DATASET_PATH,
      extract=True,
      cache_dir='.', cache_subdir='data')