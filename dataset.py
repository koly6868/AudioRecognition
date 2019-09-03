import os
import numpy as np
import pandas as pd
import random    
from glob import glob
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from noize import make_noise

class DatasetGenerator():
    def __init__(self, label_set, 
                 sample_rate=16000):
        
        self.label_set = label_set
        self.sample_rate = sample_rate
            
    # Covert string to numerical classes              
    def index_of_label(self, label):
        return self.label_set.index(label)
    
    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        return self.label_set[labels]               
        

    def load_data(self, DIR):

        # Get all paths inside DIR that ends with wav
        wav_files_for_each_word = list(map(lambda l: os.listdir(DIR + r"/" + l),self.label_set)) #glob(os.path.join(DIR, '*/*wav'))       
        
        # Loop over files to get samples
        data = []
        for label_id, wav_files in enumerate(wav_files_for_each_word):
            for name in wav_files:
                if name[-4:] == ".wav":              
                    label = self.label_set[label_id]
                    #IMPROVE
                    path = DIR + r"/" + label + r"/" + name 
                    sample = (label, label_id, name, path)
                    data.append(sample)
            
        # Data Frames with samples' labels and paths     
        df = pd.DataFrame(data, columns = ['label', 'label_id', 'user_id', 'wav_file'])
        
        self.df = df
        
        return self.df

    def apply_train_test_split(self, test_size, random_state):
        
        self.df_train, self.df_test = train_test_split(self.df, 
                                                       test_size=test_size,
                                                       random_state=random_state)
        
    def apply_train_val_split(self, val_size, random_state):
        
        self.df_train, self.df_val = train_test_split(self.df_train, 
                                                      test_size=val_size, 
                                                      random_state=random_state)
        
    def read_wav_file(self, x):
        # Read wavfile using scipy wavfile.read
        _, wav = wavfile.read(x) 
        # Normalize
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            
        return wav
    
    def process_wav_file(self, x, threshold_freq=5500, eps=1e-10):
        # Read wav file to array
        wav = self.read_wav_file(x)
        #wav = make_noise(wav)
        # Sample rate
        L = self.sample_rate
        # If longer then randomly truncate
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i+L)]  
        # If shorter then randomly add silence
        elif len(wav) < L:
            rem_len = L - len(wav)
            silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
            j = np.random.randint(0, rem_len)
            silence_part_left  = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
        # Create spectrogram using discrete FFT (change basis to frequencies)
        freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        # Cut high frequencies 
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq,:]
            freqs = freqs[freqs <= threshold_freq]
        # Log spectrogram
        amp = np.log(np.abs(spec)+eps)
    
        return np.expand_dims(amp, axis=2) 

    def generator(self, batch_size, mode):
        while True:
            # Depending on mode select DataFrame with paths
            if mode == 'train':
                df = self.df_train 
                ids = random.sample(range(df.shape[0]), df.shape[0])
            elif mode == 'val':
                df = self.df_val
                ids = list(range(df.shape[0]))
            elif mode == 'test':
                df = self.df_test
                ids = list(range(df.shape[0]))
            else:
                raise ValueError('The mode should be either train, val or test.')
                
            # Create batches (for training data the batches are randomly permuted)
            for start in range(0, len(ids), batch_size):
                X_batch = []
                if mode != 'test': 
                    y_batch = []
                end = min(start + batch_size, len(ids))
                i_batch = ids[start:end]
                for i in i_batch:
                    X_batch.append(self.process_wav_file(df.wav_file.values[i]))
                    if mode != 'test':
                        y_batch.append(df.label_id.values[i])
                X_batch = np.array(X_batch)

                if mode != 'test':
                    y_batch = to_categorical(y_batch, num_classes = len(self.label_set))
                    yield (X_batch, y_batch)
                else:
                    yield X_batch
    
    def get_val_data(self,path):

        words = os.listdir(path)
        for i,word in enumerate(words):
            paths = [os.path.join(path,word,file) for file in os.listdir(os.path.join(path,word))]
            x = np.array([self.process_wav_file(path) for path in paths])
            y = [self.label_set.index(word) for i in range(x.shape[0])]
            if i == 0:
                x_val = x
                y_val = y
            else:
                x_val = np.concatenate((x_val,x),axis=0)
                y_val = np.concatenate((y_val,y),axis=0)
        y_val = to_categorical(y_val, num_classes = len(self.label_set))

        return x_val, y_val




