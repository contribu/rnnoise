#!/usr/bin/python

from __future__ import print_function

import datetime
import math
import os

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))

def my_safecrossentropy(y_pred, y_true):
    return K.binary_crossentropy(0.1 + 0.8 * y_pred, 0.1 + 0.8 * y_true)

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * my_safecrossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*my_safecrossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

class MySequence(keras.utils.Sequence):
    def __init__(self, x_train, y_train, vad_train, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.vad_train = vad_train
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_x = self.x_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_vad = self.vad_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, [batch_y, batch_vad]

    def __len__(self):
        return math.ceil(len(self.x_train) / self.batch_size)

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
gru_activation = 'tanh'
main_input = Input(shape=(None, 42), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(24, activation=gru_activation, recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation=gru_activation, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

denoise_gru = GRU(96, activation=gru_activation, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True)

model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer=optimizer, loss_weights=[10, 0.5])

batch_size = 32

print('Loading data...')
with h5py.File('denoise_data9.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')

window_size = 2000

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :42]
x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

all_data = 0;
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

print('Train...')
dir = datetime.datetime.now().strftime("train%Y%m%d_%H%M%S")
os.makedirs(os.path.join(dir), exist_ok=True)
modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath = os.path.join(dir, 'weights.{epoch:03d}-{val_loss:.2f}.hdf5'),
                                                  monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='min',
                                                  period=1)
# model.fit(x_train, [y_train, vad_train],
#           batch_size=batch_size,
#           epochs=120,
#           validation_split=0.1,
#           callbacks=[modelCheckpoint])

x_train_train, x_val, y_train_train, y_val, vad_train_train, vad_val = train_test_split(x_train, y_train, vad_train, test_size=0.1, shuffle=True, random_state=1)

train_gen = MySequence(x_train_train, y_train_train, vad_train_train, batch_size)
val_gen = MySequence(x_val, y_val, vad_val, batch_size)
model.fit_generator(
    generator=train_gen,
    epochs=120,
    steps_per_epoch=len(train_gen),
    verbose=1,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    callbacks=[modelCheckpoint])

model.save("newweights9i.hdf5")


