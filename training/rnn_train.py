#!/usr/bin/python

from __future__ import print_function

import argparse
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
from keras.layers import BatchNormalization
from keras.layers import CuDNNGRU
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

parser = argparse.ArgumentParser(description='train script')
parser.add_argument('--data', default='denoise_data9.h5')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--reg', type=float, default=0.000001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--cudnngru', action='store_true')
args = parser.parse_args()

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

reg = args.reg
constraint = WeightClip(0.499)

def create_gru(units, name):
    if args.cudnngru:
        result = CuDNNGRU(units, return_sequences=True, name=name, kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)
    else:
        # return GRU(units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name=name, kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)
        result = GRU(units, recurrent_activation='sigmoid', reset_after=True, return_sequences=True, name=name, kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)
    return result

print('Build model...')
print(args)

# gru_lrelu_alpha = 1.0 / 5.5 # from "Empirical Evaluation of Rectified Activations in Convolution Network"

main_input = Input(shape=(None, 42), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
if args.dropout > 0:
    tmp = Dropout(args.dropout)(tmp)

vad_gru = create_gru(24, 'vad_gru')(tmp)
if args.dropout > 0:
    vad_gru = Dropout(args.dropout)(vad_gru)
# vad_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="vad_gru_activation")(GRU(24, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp))
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = create_gru(48, 'noise_gru')(noise_input)
if args.dropout > 0:
    noise_gru = Dropout(args.dropout)(noise_gru)
# noise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="noise_gru_activation")(GRU(48, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input))

denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
denoise_gru = create_gru(96, 'denoise_gru')(noise_input)
if args.dropout > 0:
    noise_gru = Dropout(args.dropout)(noise_gru)
# denoise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="denoise_gru_activation")(GRU(96, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input))

denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True)

model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer=optimizer, loss_weights=[10, 0.5])

batch_size = args.batch_size

print('Loading data...')
with h5py.File(args.data, 'r') as hf:
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
model.fit(x_train, [y_train, vad_train],
          batch_size=batch_size,
          epochs=120,
          validation_split=0.1,
          callbacks=[modelCheckpoint])

# x_train_train, x_val, y_train_train, y_val, vad_train_train, vad_val = train_test_split(x_train, y_train, vad_train, test_size=0.1, shuffle=True, random_state=1)
#
# train_gen = MySequence(x_train_train, y_train_train, vad_train_train, batch_size)
# val_gen = MySequence(x_val, y_val, vad_val, batch_size)
# model.fit_generator(
#     generator=train_gen,
#     epochs=120,
#     steps_per_epoch=len(train_gen),
#     verbose=1,
#     validation_data=val_gen,
#     validation_steps=len(val_gen),
#     callbacks=[modelCheckpoint])

model.save("newweights9i.hdf5")


