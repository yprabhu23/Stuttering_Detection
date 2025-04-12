import tensorflow as tf
import tensorflow_io as tfio
from IPython.display import Audio
import matplotlib.pyplot as plt


# audio = tfio.audio.AudioIOTensor('gs://cloud-samples-tests/speech/brooklyn.flac')
#
# print(audio)
#
# tf.convert_to_tensor(audio, tf.float32)


audio = tfio.audio.AudioIOTensor('/home/yashprabhu/Downloads/ml-stuttering-events-dataset-main/[CLIP_DIR]/clips/HeStutters/0/HeStutters_0_0.wav', dtype=tf.int16)
print(audio)
audio_tensor = tf.cast(tf.squeeze(audio.to_tensor()), tf.float32)


plt.figure()
plt.plot(audio_tensor.numpy())
plt.show()
