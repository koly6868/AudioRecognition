import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os
import uuid

SAMPLE_RATE = 16000
DURATION = 1.2
NOIZE_COFFICIENT = 1
DIR = "people"


print("start")
arr = sd.rec(frames=int(SAMPLE_RATE*DURATION),samplerate=SAMPLE_RATE,channels=1,blocking=True)
treshold = np.amax(arr[:int(SAMPLE_RATE*0.2)])
arr = np.where(np.abs(arr) < treshold, arr/NOIZE_COFFICIENT, arr)

word = input("enter word\n")
dir_path = os.path.join(DIR,word)
try:
    os.mkdir(dir_path)
except Exception:
    pass

name = "0"

file_path = os.path.join(dir_path, str(uuid.uuid4()) + ".wav")
write(file_path,SAMPLE_RATE,arr)
#sd.play(arr[-16000:],samplerate=16000, blocking=True)