import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from dataset import DatasetGenerator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Dense, Flatten, Dropout
from keras.models import Model, load_model
from dataset import DatasetGenerator
from keras.utils import plot_model
from keras.optimizers import rmsprop
import os
import vizualizer as vz
import pickle
#####################
class LossHistory(Callback):

    def __init__(self, logs_dir):
        self.logs = []
        self.logs_dir = logs_dir

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        with open(os.path.join(self.logs_dir,str(epoch)),mode="w+b") as file:
            pickle.dump(self.logs,file)
        
 
###########

CHECPOINT_DIR = "models"
LOGS_DIR = "models/logs"
TRAIN_DIR = "input"
VAL_DIR = "people"
INPUT_SHAPE = (177,98,1)
BATCH = 32
EPOCHS = 20
LABELS = list(filter(lambda e: e[0] != "_", os.listdir('input')))
NUM_CLASSES = len(LABELS)
NETWORK_NAME = "model.hdf5"
dsGen = DatasetGenerator(label_set=LABELS) 
# Load DataFrame with paths/labels 
df = dsGen.load_data(TRAIN_DIR)
dsGen.apply_train_test_split(test_size=0.3, random_state=2018)
#dsGen.apply_train_val_split(val_size=0.2, random_state=2018)

def deep_cnn(features_shape, num_classes, act='relu'):
 
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape, kernel_initializer="random_uniform")(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)
    
    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv', kernel_initializer="random_uniform")(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)
 
    # Block 3
    o = Conv2D(64, (3, 3), activation=act, padding='same', strides=1, name='block3_conv', kernel_initializer="random_uniform")(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)
 
    # Flatten
    o = Flatten(name='flatten')(o)
    
    # Dense layer
    o = Dense(512, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(o)
    o = Dropout(0.2, name='dropout')(o)
    
    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)
 
    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return Model(inputs=x, outputs=o)



def train(model):
    #callbacks = [EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')]
    callbacks = [ModelCheckpoint(os.path.join(CHECPOINT_DIR,"acc_{val_acc:.2f}--epoch_{epoch}.hdf5")),
                LossHistory(LOGS_DIR),ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)]
    history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                                steps_per_epoch=int(np.ceil(len(dsGen.df_train)/BATCH)),
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=dsGen.get_val_data(VAL_DIR) #dsGen.generator(BATCH, mode='val'),
                                )#validation_steps=int(np.ceil(len(dsGen.df_val)/BATCH)))
    return history

def test(model):
    y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'), 
                                        int(np.ceil(len(dsGen.df_test)/BATCH)), 
                                        verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = dsGen.df_test['label_id'].values
    acc_score = accuracy_score(y_true, y_pred)
    return acc_score

if os.listdir("./").count(NETWORK_NAME) == 0:
    model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
    opt = 'Adam'
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
else:
    model = load_model(NETWORK_NAME)

plot_model(model, to_file=NETWORK_NAME+".png", show_shapes=True)
vz.show_history(train(model))
print(test(model))

model.save(NETWORK_NAME)

