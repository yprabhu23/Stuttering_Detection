import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np

from pyaudio import PyAudio, paFloat32
from math import floor
from time import perf_counter as clock

import sys


_CHANNELS = 2
_RATE = 44100
bufferSize = 1024

device = PyAudio()

try:
  streamIn = device.open(
    format=paFloat32,
    channels=_CHANNELS,
    rate=_RATE,
    input=True,
    frames_per_buffer=bufferSize
  )

  streamOut = device.open(
    format=paFloat32,
    channels=_CHANNELS,
    rate=_RATE,
    output=True,
    frames_per_buffer=bufferSize
  )

except OSError as e:
  print('Error', 'No input/output device found! Connect and rerun.')
  exit()

while streamIn.is_active():
  start = clock()
  audioData = streamIn.read(bufferSize)
  streamOut.write(audioData)
  actualDelay = clock() - start

  print("lolol")