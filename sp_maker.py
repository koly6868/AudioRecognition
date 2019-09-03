from dataset import DatasetGenerator
import os
import PIL
import matplotlib.pyplot as plt
import random as rd
from scipy.io import wavfile

DIR = "input"
LABELS = list(filter(lambda e: e[0] != "_", os.listdir(DIR)))
dsGen = DatasetGenerator(LABELS)

#data = os.listdir(DIR+r"/four")
#data = data[rd.randrange(0,len(data))]
#path = DIR+r"/four/"+data
path = "test.wav"
im = dsGen.process_wav_file(path)
fig = plt.figure()    
ax = fig.add_subplot(2,1,1)
ax.imshow(im[:,:,0], aspect='auto', origin='lower')
ax.set_title('Spectrogram')
ax.set_ylabel('Freqs in Hz')
ax.set_xlabel('Seconds')

ax = fig.add_subplot(2,1,2)
wav = wavfile.read(path)[1]
ax.plot(wav)
plt.show()
print(wav.shape)