#!/usr/bin/env python3

import wave
import sys
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from vosk import Model, KaldiRecognizer, SetLogLevel

# You can set log level to -1 to disable debug messages
SetLogLevel(0)

wf = wave.open(sys.argv[1], "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)

model = Model(lang="en-us")

# You can also init model by name or with a folder path
# model = Model(model_name="vosk-model-en-us-0.21")
# model = Model("models/en")

rec = KaldiRecognizer(model, wf.getframerate())
print('sampling rate = {}'.format(wf.getframerate()))
rec.SetWords(False)
rec.SetPartialWords(False)
nframes = wf.getnframes()
print('num of frames: {}'.format(nframes))
text = ' '

while True:
    data = wf.readframes(1000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)
    text = text = rec.PartialResult()
    if 'one' in text:
        rec.Result()
        print('find {}: {}'.format('one', text))
    # else:
        # print('rec Partial result: {}'.format(text))

# print('final result: {}'.format(rec.FinalResult()))





fs, wave = wavfile.read(sys.argv[1])
f, t, S = signal.spectrogram(wave, fs)
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.linspace(0, nframes/fs, nframes), wave)
plt.subplot(1,2,2)
plt.pcolormesh(t, f, np.log(S+1), shading='gouraud', cmap='jet')

plt.show()