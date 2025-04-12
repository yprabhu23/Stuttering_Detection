import pathlib
import pandas as pd
import tensorflow as tf
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    print(parts)
    print("HIIIII")
    print(parts[-2])
    return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label


DATASET_PATH = '/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/clip_dir/clips/HeStutters/0'

data_dir = pathlib.Path(DATASET_PATH)

LABELS_PATH = '/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv'

csvreader = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
#
# thing = csvreader[csvreader["EpId"] ==0]
# print(thing)
#
# print(thing[thing["Show"] == "HeStutters"])
# print(thing(0))


headers = csvreader.columns[5:]

# namecol = pd.read_csv(LABELS_PATH, usecols=[headers])
# print(namecol["ClipID"])


print(headers)

filenames = tf.io.gfile.glob(str(data_dir) + '/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)

print('Example file tensor:', filenames[0])

test_file = tf.io.read_file(DATASET_PATH + '/HeStutters_0_0.wav')
print(get_label(DATASET_PATH + '/HeStutters_0_0.wav'))

test_audio, _ = tf.audio.decode_wav(contents=test_file)
print(test_audio.shape)


AUTOTUNE = tf.data.AUTOTUNE

files_ds = filenames#tf.data.Dataset.from_tensor_slices(train_files)

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()
