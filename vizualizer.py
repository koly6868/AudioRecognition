import matplotlib.pyplot as plt
from scipy.io import wavfile
from dataset import DatasetGenerator
import os

DIR = "input"
LABELS = list(filter(lambda e: e[0] != "_", os.listdir(DIR)))
dsGen = DatasetGenerator(LABELS)

def show_wavfile(path):
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

def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy') 
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    dir_path = r"input/right"
    paths = os.listdir(dir_path)
    paths = paths[:2]
    paths = [os.path.join(dir_path,path) for path in paths]
    for path in paths:
        show_wavfile(path)