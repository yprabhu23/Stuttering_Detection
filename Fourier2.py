import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
from keras.layers import LSTM

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.

arrayThing = ['Block', 'SoundRep','WordRep','Interjection','NoStutteredWords']
# arrayThing = ['Block','DifficultToUnderstand','Interjection','NaturalPause','NoStutteredWords','Prolongation','SoundRep', 'WordRep']
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# DATASET_PATH = '/media/yashprabhu/Extreme SSD/SortedBy'
# data_dir = pathlib.Path(DATASET_PATH)
# csvreader = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
# commands = commands[commands != 'README.md']
# print('Commands:', commands)

# filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
# filenames = tf.random.shuffle(filenames)
# num_samples = len(filenames)
# train_files = filenames[:20692]
# print(train_files)
# val_files = filenames[20692:20692+2586]
# test_files = filenames[-2586:]
# print('Number of total examples:', num_samples)
#
# print('Example file tensor:', filenames[0])


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


# def get_label_2(file_path):
#     print("this jawn")


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
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return output_ds

for thing in arrayThing:
    print(thing)
    TOTAL_TRUE_POSITIVES = 0
    TOTAL_TRUE_NEGATIVES = 0
    TOTAL_FALSE_POSITIVES = 0
    TOTAL_FALSE_NEGATIVES = 0
    # for z in range(0, 5):
    DATASET_PATH = '/media/yashprabhu/Extreme SSD/SortedBy' + thing
    data_dir = pathlib.Path(DATASET_PATH)
    csvreader = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']


    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    train_files = filenames[:18106]

    val_files = filenames[18106:18106+ 2587]
    test_files = filenames[-5174:]




    AUTOTUNE = tf.data.AUTOTUNE

    files_ds = tf.data.Dataset.from_tensor_slices(filenames)  # trainingdata

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

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()

    spectrogram_ds = waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(spectrogram.numpy(), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')

    plt.show()

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    # val_ds = preprocess_dataset(train_ds)
    # test_ds = preprocess_dataset(train_ds)
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape1 = spectrogram.shape

    num_labels = len(commands)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))


    model = models.Sequential([
        layers.Input(shape=input_shape1),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        # layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),


        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    tf.keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    model.summary()


    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    # train_ds.reshape(None,124,129,1)filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    train_files = filenames
    val_files = filenames
    test_files = filenames

    EPOCHS = 200


    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    history = model.fit(
        train_ds,
        validation_data=train_ds,
        epochs=EPOCHS,
        callbacks=tensorboard_callback,  # [tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)



    y_true = test_labels

    counter = 0
    TRUE_POSITIVES = 0
    TRUE_NEGATIVES = 0
    FALSE_POSITIVES = 0
    FALSE_NEGATIVES = 0
    ones = 0
    zeros = 0
    for i in range (0,len(y_pred)):

        if(y_pred[i]==y_true[i]):
            if (y_pred[i] == 0):
                TRUE_POSITIVES += 1
            else:
                TRUE_NEGATIVES += 1
        else:
            if (y_pred[i] == 0):
                FALSE_POSITIVES += 1
            else:
                FALSE_NEGATIVES += 1
    accuracy = (TRUE_POSITIVES + TRUE_NEGATIVES) / (TRUE_POSITIVES + TRUE_NEGATIVES + FALSE_NEGATIVES + FALSE_POSITIVES)
    precision = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES))
    recall = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES))
    F1 = 2*(precision*recall)/(precision+recall)

    print()
    print("accuracy: ",accuracy)
    print()

    print("precision: ",precision)
    print("recall: ",recall)
    print("true positives",TRUE_POSITIVES)
    print("true negatives",TRUE_NEGATIVES)
    print("false positives",FALSE_POSITIVES)
    print("false negatives",FALSE_NEGATIVES)
    print("F1 score: ", F1)


    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=commands,
                yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    TRUE_POSITIVES = 0
    TRUE_NEGATIVES = 0
    FALSE_POSITIVES = 0
    FALSE_NEGATIVES = 0
    print("NEXT")
#     df = pd.read_csv('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/SEP-28k_labels.csv')
#
#     # DATASET_PATH = '/home/yashprabhu/Documents/SortedByProlongation'
#     # data_dir = pathlib.Path(DATASET_PATH)
#
#     # commands = np.array(tf.io.gfile.listdir(str(data_dir)))
#     # commands = commands[commands != 'README.md']
#     for file in test_files:
#
#         filename = str(df.Show[index]) + '_' + str(df.EpId[index]) + '_' + str(df.ClipId[index]) + str(
#             '.wav')  # get the filename
#         directory = '/home/yashprabhu/Documents/Sep28k/clips/HVSA/'
#         sample_file = directory + str(df.EpId[index]) + '/' + str(filename)
#
#         sample_ds = preprocess_dataset([str(sample_file)])
#
#         for spectrogram in sample_ds.batch(1):
#             ActuallyProlonged = False
#             PredictedProlonged = False
#             if (df.iloc[index][arrayThing[z]]) > 0:
#                 ActuallyProlonged = True
#             else:
#                 ActuallyProlonged = False
#
#             startTime = time.time()
#
#             prediction = model.predict(spectrogram)
#
#             endTime = time.time() - startTime
#             sumTime += endTime
#             counter += 1
#
#             if prediction[0][0] > prediction[0][1]:
#                 PredictedProlonged = False
#             else:
#                 PredictedProlonged = True
#
#             if PredictedProlonged == True and ActuallyProlonged == True:
#                 TRUE_POSITIVES += 1
#             elif PredictedProlonged == True and ActuallyProlonged == False:
#                 FALSE_POSITIVES += 1
#             elif PredictedProlonged == False and ActuallyProlonged == False:
#                 TRUE_NEGATIVES += 1
#             else:
#                 FALSE_NEGATIVES += 1
#
#         # prediction = model(spectrogram)
#         # plt.bar(commands, tf.nn.softmax(prediction[0]))
#         # plt.title(f'Predictions for "{commands[label[0]]}"')
#         # plt.show()
#     print("True Positives", TRUE_POSITIVES)
#     print("True Negatives", TRUE_NEGATIVES)
#     print("False Positives", FALSE_POSITIVES)
#     print("False Negatives", FALSE_NEGATIVES)
#     print()
#     accuracy = (TRUE_POSITIVES + TRUE_NEGATIVES) / (TRUE_POSITIVES + TRUE_NEGATIVES + FALSE_NEGATIVES + FALSE_POSITIVES)
#     precision = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES))
#     recall = (TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES))
#
#     print()
#     print(accuracy)
#     print(precision)
#     print(recall)
#
# print(sumTime / counter)
#
#     # sample_file = data_dir / 'Prolonged/HeStutters_0_31.wav'
#     #
#     # sample_ds = preprocess_dataset([str(sample_file)])
#     #
#     # for spectrogram, label in sample_ds.batch(1):
#     #     prediction = model(spectrogram)
#     #     plt.bar(commands, tf.nn.softmax(prediction[0]))
#     #     plt.title(f'Predictions for "{commands[label[0]]}"')
#     #     plt.show()
#
#     model.save('/home/yashprabhu/Documents/GitHub/AudioThings/modelGit/TESTING/' + str(arrayThing[z]))
