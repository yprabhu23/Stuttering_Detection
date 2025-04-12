import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from keras.layers import LSTM

from tensorflow.keras import layers
from tensorflow.keras import models

# model = keras.models.load_model('/home/yashprabhu/Documents/GitHub/AudioThings/modelGit/LSTM/Block/')

arrayThing = [ 'Prolongation']#'Block', 'DifficultToUnderstand', 'Interjection', 'NaturalPause', 'NoStutteredWords', 'Prolongation','SoundRep',
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# DATASET_PATH = '/home/yashprabhu/Documents/SortedByProlongation'
# data_dir = pathlib.Path(DATASET_PATH)
df = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
# commands = commands[commands != 'README.md']


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
    return parts[-2]


def get_label_2(file_path):
    print("this jawn")


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    # label_id = tf.argmax(label == commands)
    return spectrogram


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return output_ds


AUTOTUNE = tf.data.AUTOTUNE
TRUE_POSITIVES = 0
TRUE_NEGATIVES = 0
FALSE_POSITIVES = 0
FALSE_NEGATIVES = 0
# index = 33
sumTime = 0
counter = 0
for z in range(0, len(arrayThing)):
    TRUE_POSITIVES = 0
    TRUE_NEGATIVES = 0
    FALSE_POSITIVES = 0
    FALSE_NEGATIVES = 0
    model = keras.models.load_model('/home/yashprabhu/Documents/GitHub/AudioThings/modelGit/TESTING/' +str(arrayThing[z])+'/')
    print(arrayThing[z], ":")
    # DATASET_PATH = '/home/yashprabhu/Documents/SortedByProlongation'
    # data_dir = pathlib.Path(DATASET_PATH)
    df = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
    # commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    # commands = commands[commands != 'README.md']
    for index in range(3686,4420):
        filename = str(df.Show[index]) + '_' + str(df.EpId[index]) + '_' + str(df.ClipId[index]) + str(
            '.wav')  # get the filename
        directory = '/home/yashprabhu/Documents/Sep28k/clips/HVSA/'
        sample_file = directory + str(df.EpId[index]) + '/' + str(filename)

        sample_ds = preprocess_dataset([str(sample_file)])

        for spectrogram in sample_ds.batch(1):
            ActuallyProlonged = False
            PredictedProlonged = False
            if (df.iloc[index][arrayThing[z]]) > 0:
                ActuallyProlonged = True
            else:
                ActuallyProlonged = False


            startTime = time.time()

            prediction = model.predict(spectrogram)

            endTime=time.time()-startTime
            sumTime+=endTime
            counter+=1

            if prediction[0][0] > prediction[0][1]:
                PredictedProlonged = False
            else:
                PredictedProlonged = True

            if PredictedProlonged==True and ActuallyProlonged == True:
                TRUE_POSITIVES+=1
            elif PredictedProlonged==True and ActuallyProlonged==False:
                FALSE_POSITIVES+=1
            elif PredictedProlonged ==False and ActuallyProlonged == False:
                TRUE_NEGATIVES+=1
            else:
                FALSE_NEGATIVES+=1

        # prediction = model(spectrogram)
        # plt.bar(commands, tf.nn.softmax(prediction[0]))
        # plt.title(f'Predictions for "{commands[label[0]]}"')
        # plt.show()
    print("True Positives" , TRUE_POSITIVES)
    print("True Negatives" , TRUE_NEGATIVES)
    print("False Positives" , FALSE_POSITIVES)
    print("False Negatives", FALSE_NEGATIVES)
    print()
    accuracy = (TRUE_POSITIVES + TRUE_NEGATIVES) / (TRUE_POSITIVES + TRUE_NEGATIVES + FALSE_NEGATIVES + FALSE_POSITIVES)
    precision = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES))
    recall = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES))

    print()
    print(accuracy)
    print(precision)
    print(recall)

print(sumTime/counter)

