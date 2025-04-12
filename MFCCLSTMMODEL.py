import Padding as padding
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
import pathlib
import numpy as np
import librosa.display

DATASET_PATH = '/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/clips/'
data_dir = DATASET_PATH
df = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')

sample_num = 2  # pick a file to display
filename = str(df.Show[sample_num]) + '_' + str(df.EpId[sample_num]) + '_' + str(df.ClipId[sample_num]) + str(
    '.wav')  # get the filename
print(filename)
location = data_dir + str(df.Show[sample_num]) + '/' + str(df.EpId[sample_num]) + '/' + str(filename)
print(location)
y, sr = librosa.load(data_dir + str(df.Show[sample_num]) + '/' + str(df.EpId[sample_num]) + '/' + str(filename))
print(y)
print(sr)
# librosa.display.waveshow(y,sr=sr, x_axis='time', color='purple',offset=0.0)

hop_length = 512  # the default spacing between frames
n_fft = 255  # number of samples
# cut the sample to the relevant times
y_cut = y
MFCCs = librosa.feature.mfcc(y_cut, n_fft=n_fft, hop_length=hop_length, n_mfcc=128)
fig, ax = plt.subplots(figsize=(20, 7))
librosa.display.specshow(MFCCs, sr=sr, cmap='cool', hop_length=hop_length)
ax.set_xlabel('Time', fontsize=15)
ax.set_title('MFCC', size=20)
plt.colorbar()
plt.show()

print(df.iloc[33]["Prolongation"])


def get_features(df_in):
    features = []  # list to save features
    labels = []  # list to save labels
    for index in range(0, df_in):
        # get the filename
        filename = str(df.Show[sample_num]) + '_' + str(df.EpId[sample_num]) + '_' + str(df.ClipId[sample_num]) + str(
            '.wav')
        # cut to start of signal
        # tstart = df_in.iloc[index]['t_min']
        # #cut to end of signal
        # tend = df_in.iloc[index]['t_max']
        # save labels
        stutter_id = 0
        prolongs = df.iloc[index]['Prolongation']
        if prolongs > 0:
            stutter_id = 1
        else:
            stutter_id = 0

        # load the file

        y, sr = librosa.load(data_dir + str(df.Show[sample_num]) + '/' + str(df.EpId[sample_num]) + '/' + filename,
                             sr=28000)
        # cut the file from tstart to tend
        y_cut = y
        data = np.array([(librosa.feature.mfcc(y_cut,
                                               n_fft=n_fft, hop_length=hop_length, n_mfcc=128), 1, 400)])
        features.append(data)
        labels.append(stutter_id)
    output = np.concatenate(features, axis=0)
    return (np.array(output), labels)


X, y = get_features(40)


X = np.array(X)
# X = np.array((X - np.min(X))) / (np.max(X) - np.min(X))
# X = X / np.std(X)
y = np.array(y)

# Split twice to get the validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
# Print the shapes
print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))


input_shape=(128,1000)
model = tf.keras.Sequential()
model.add(LSTM(128,input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(24, activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['acc'])
X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_val = np.asarray(X_val).astype('float32')
X_val = np.asarray(y_val).astype('float32')

history = model.fit(X_train, y_train, epochs=50, batch_size=72,
                    validation_data=(X_val, y_val), shuffle=False)
