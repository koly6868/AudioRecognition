from keras.models import load_model
from dataset import DatasetGenerator
import os
import random as rd
import numpy as np
import vizualizer as vz

LABELS_DIR = "input"
DIR = 'people'
LABELS = list(filter(lambda e: e[0] != "_", os.listdir(LABELS_DIR)))
PROBABILITY = 0.86
dsGen = DatasetGenerator(label_set=LABELS) 
#data = os.listdir(DIR+r"/bed")
#data = data[rd.randrange(0,len(data))]
#path = DIR+r"/bed/"+data
word = "tree"
#[vz.show_wavfile(os.path.join(DIR,word,file)) for file in os.listdir(os.path.join(DIR,word))]
paths = [os.path.join(DIR,word,file) for file in os.listdir(os.path.join(DIR,word))]
data = np.array([dsGen.process_wav_file(path) for path in paths])
model = load_model("model.hdf5")
answ = [LABELS.index(word) for i in range(data.shape[0])] 

def test(model,data,answ, verbouse=1):
    pred = model.predict(data,batch_size=data.shape[0])
    err_indexes = []
    for i, pred in enumerate(pred):
        max_ind = np.argmax(pred)
        if verbouse == 1:
            print(LABELS[max_ind], pred[max_ind], pred[answ[i]])
        if np.argmax(pred) != answ[i]:
            err_indexes.append(i)
    return (len(answ) - len(err_indexes))/len(answ), err_indexes 

acc, err_indexes = test(model,data,answ)
#print(acc)
#if len(err_indexes) != 0:
#    print(np.vectorize(lambda i: paths[i])(err_indexes))

def global_test(path, model):
    err_count = 0
    lenght = 0
    words = os.listdir(path)
    for word in words:
        paths = [os.path.join(path,word,file) for file in os.listdir(os.path.join(path,word))]
        data = np.array([dsGen.process_wav_file(path) for path in paths])
        answ = [LABELS.index(word) for i in range(data.shape[0])]
        _, err_ind = test(model,data,answ,verbouse=0)
        err_count += len(err_ind)
        lenght += len(answ)
    
    return (lenght - err_count)/lenght

print(global_test(DIR,model))

    
