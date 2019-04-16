#!/usr/bin/python

from __future__ import print_function

import argparse
import datetime
import math
import os
import sys
import importlib
import random

from sklearn.model_selection import train_test_split

import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
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
from keras.utils import multi_gpu_model

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

from tcn import TCN
# import autokeras as ak

# import mixup_generator from .py
# mixup_generator = importlib.machinery.SourceFileLoader('mixup_generator', os.path.join(os.path.dirname(__file__), '../deps/mixup-generator/mixup_generator.py')).load_module()

# dump_to_simple_cpp = importlib.machinery.SourceFileLoader('dump_to_simple_cpp', os.path.join(os.path.dirname(__file__), '../deps/keras2cpp/dump_to_simple_cpp_custom.py')).load_module()
# pt = importlib.machinery.SourceFileLoader('pt', os.path.join(os.path.dirname(__file__), '../deps/pocket-tensor/pt.py')).load_module()

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
parser.add_argument('--input_dropout', type=float, default=0.0)
parser.add_argument('--hidden_units', type=float, default=1.0)
parser.add_argument('--cudnngru', action='store_true')
parser.add_argument('--mmap', action='store_true')
parser.add_argument('--mixup', type=int, default=0)
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--safe_crossentropy_factor', type=float, default=0.0)
parser.add_argument('--loss_type', default='original')
parser.add_argument('--log_loss_bias', type=float, default='1e-7')
parser.add_argument('--arch', default='original')
parser.add_argument('--window_size', type=int, default=2000)
parser.add_argument('--window_overlap', type=int, default=0)
parser.add_argument('--noise_prob', type=float, default=0.0)
parser.add_argument('--noise_stddev', type=float, default=10.0)
parser.add_argument('--tcn_layers', type=int, default=3)
parser.add_argument('--tcn_dilation_order', type=int, default=5)
parser.add_argument('--bands', type=int, default=22)
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--resume', default='')
args = parser.parse_args()

def my_safecrossentropy(y_pred, y_true):
    f = args.safe_crossentropy_factor
    return K.binary_crossentropy(f + (1.0 - 2 * f) * y_pred, f + (1.0 - 2 * f) * y_true)

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * my_safecrossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    if args.loss_type == 'log':
        return K.mean(mymask(y_true) * K.square(K.log(args.log_loss_bias + y_pred) - K.log(args.log_loss_bias + K.abs(y_true))), axis=-1)
    elif args.loss_type == 'original':
        return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*my_safecrossentropy(y_pred, y_true)), axis=-1)
    else:
        raise Exception('unknown loss_type')

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2,name=''):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

class MyMixupGenerator:
    def __init__(self, x_train, y_train, vad_train, batch_size, alpha):
        self.x_train = x_train
        self.y_train = y_train
        self.vad_train = vad_train
        self.batch_size = batch_size
        self.alpha = alpha

    def __call__(self):
        while True:
            idx1 = random.randrange(0, len(self.x_train) // self.batch_size)
            idx2 = random.randrange(0, len(self.x_train) // self.batch_size)
            batch_x1 = self.x_train[idx1 * self.batch_size:(idx1 + 1) * self.batch_size]
            batch_y1 = self.y_train[idx1 * self.batch_size:(idx1 + 1) * self.batch_size]
            batch_vad1 = self.vad_train[idx1 * self.batch_size:(idx1 + 1) * self.batch_size]

            batch_x2 = self.x_train[idx2 * self.batch_size:(idx2 + 1) * self.batch_size]
            batch_y2 = self.y_train[idx2 * self.batch_size:(idx2 + 1) * self.batch_size]
            batch_vad2 = self.vad_train[idx2 * self.batch_size:(idx2 + 1) * self.batch_size]

            lam = np.random.beta(self.alpha, self.alpha)
            yield batch_x1 * (1 - lam) + batch_x2 * lam, [batch_y1 * (1 - lam) + batch_y2 * lam, batch_vad1 * (1 - lam) + batch_vad2 * lam]

    def __len__(self):
        return math.ceil(len(self.x_train) / self.batch_size)

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

class MyLazySequence(keras.utils.Sequence):
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

window_size = args.window_size
feature_count = args.bands + 20

# https://deepage.net/deep_learning/2016/11/30/resnet.html
def double_cnn_block(unit, conv_shape, input):
    # wide res net
    x = input
    x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.Activation('elu')(x)
    x = Conv2D(unit, conv_shape, padding='same', use_bias=False)(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.Activation('elu')(x)
    if args.dropout > 0:
        x = Dropout(args.dropout)(x)
    x = Conv2D(unit, conv_shape, padding='same', use_bias=False)(x)
    return x

def res_block(unit, conv_shape, input):
    x = double_cnn_block(unit, conv_shape, input)
    x = keras.layers.Add()([x, input])
    return x

def res_block2(unit, conv_shape1, conv_shape2, input):
    x1 = double_cnn_block(unit // 2, conv_shape1, input)
    x2 = double_cnn_block(unit // 2, conv_shape2, input)
    x = keras.layers.concatenate([x1, x2])
    x = keras.layers.Add()([x, input])
    return x

def tcn_res_block(unit, kernel_size, dilation_rate, input):
    x = input
    x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.Activation('elu')(x)
    x = keras.layers.Conv1D(unit, kernel_size, dilation_rate=dilation_rate, padding='causal', use_bias=False)(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    x = keras.layers.Activation('elu')(x)
    if args.dropout > 0:
        x = Dropout(args.dropout)(x)
    x = keras.layers.Conv1D(unit, kernel_size, dilation_rate=dilation_rate, padding='causal', use_bias=False)(x)
    if input.shape[-1] != unit:
        input = Dense(unit)(input)
    x = keras.layers.Add()([x, input])
    return x

def tcn_res_blocks(unit, kernel_size, dilation_rates, input):
    x = input
    for dilation_rate in dilation_rates:
        x = tcn_res_block(unit, kernel_size, dilation_rate, x)
    return x

if args.resume != '':
    custom_objects = {
        'my_safecrossentropy': my_safecrossentropy,
        'my_crossentropy': my_crossentropy,
        'mymask': mymask,
        'msse': msse,
        'mycost': mycost,
        'my_accuracy': my_accuracy,
        'WeightClip': WeightClip
    }
    model = keras.models.load_model(args.resume, custom_objects=custom_objects)
elif args.arch == 'original':
    main_input = Input(shape=(None, feature_count), name='main_input')

    tmp = Dense(math.ceil(24 * args.hidden_units), activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
    if args.dropout > 0:
        tmp = Dropout(args.dropout)(tmp)

    vad_gru = create_gru(math.ceil(24 * args.hidden_units), 'vad_gru')(tmp)
    if args.dropout > 0:
        vad_gru = Dropout(args.dropout)(vad_gru)
    # vad_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="vad_gru_activation")(GRU(24, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp))
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
    noise_gru = create_gru(math.ceil(48 * args.hidden_units), 'noise_gru')(noise_input)
    if args.dropout > 0:
        noise_gru = Dropout(args.dropout)(noise_gru)
    # noise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="noise_gru_activation")(GRU(48, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input))

    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
    denoise_gru = create_gru(math.ceil(96 * args.hidden_units), 'denoise_gru')(denoise_input)
    if args.dropout > 0:
        denoise_gru = Dropout(args.dropout)(denoise_gru)
    # denoise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="denoise_gru_activation")(GRU(96, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input))

    denoise_output = Dense(args.bands, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
elif args.arch == 'cnn':
    main_input = Input(shape=(window_size, feature_count), name='main_input')
    input_dropout = Dropout(args.input_dropout)(main_input)
    reshaped = Reshape((window_size, feature_count, 1))(input_dropout)
    # conv1 = Conv2D(int(16 * args.hidden_units), (3, 3), dilation_rate=(1, 1), padding='same', use_bias=False)(reshaped)
    # conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    # conv1 = keras.layers.Activation('elu')(conv1)

    # 小さいデータセットでいろいろ手元で実験した結果
    # 周波数方向に撹拌するのは効くらしい (1, 42)など。(3, 3)だけとかにして撹拌しなくすると、かなり成績落ちる
    # http://www.jordipons.me/media/UPC-2018.pdf とも整合している
    # res_block2の数は、4 < 5 = 6らしい

    conv1 = res_block2(int(16 * args.hidden_units), (1, feature_count), (13, 3), reshaped)
    conv1 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv1)
    conv2 = res_block2(int(16 * args.hidden_units), (1, feature_count), (13, 3), conv1)
    conv2 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv2)
    conv3 = res_block2(int(16 * args.hidden_units), (1, feature_count), (13, 3), conv2)
    conv3 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv3)
    conv4 = res_block2(int(16 * args.hidden_units), (1, feature_count), (13, 3), conv3)
    conv4 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv4)
    conv5 = res_block2(int(16 * args.hidden_units), (1, feature_count), (13, 3), conv4)
    conv5 = keras.layers.AveragePooling2D(pool_size=(window_size // 16, 1), strides=None, padding='valid')(conv5)

    flatten = Flatten()(conv5);
    vad_output = Dense(1, activation='sigmoid', name='vad_output')(flatten)
    denoise_output = Dense(args.bands, activation='sigmoid', name='denoise_output')(flatten)
    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
elif args.arch == 'cnn2':
    # 初期バージョンに近い形だが、小さいデータで試したらcnnより成績悪い
    main_input = Input(shape=(window_size, feature_count), name='main_input')
    input_dropout = Dropout(args.input_dropout)(main_input)
    reshaped = Reshape((window_size, feature_count, 1))(input_dropout)

    conv1 = res_block2(int(16 * args.hidden_units), (3, 3), (3, 3), reshaped)
    conv2 = res_block2(int(16 * args.hidden_units), (1, feature_count), (window_size, 1), conv1)
    conv3 = res_block2(int(16 * args.hidden_units), (3, feature_count), (41, 3), conv2)
    conv3 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv3)
    conv4 = res_block2(int(16 * args.hidden_units), (3, feature_count), (41, 3), conv3)
    conv4 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv4)
    conv5 = res_block2(int(16 * args.hidden_units), (3, feature_count), (41, 3), conv4)
    conv5 = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(conv5)

    flatten = Flatten()(conv5);
    vad_output = Dense(1, activation='sigmoid', name='vad_output')(flatten)
    denoise_output = Dense(args.bands, activation='sigmoid', name='denoise_output')(flatten)
    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
elif args.arch == 'original_tcn':
    # originalをtcnに置き換えたもの
    main_input = Input(shape=(None, feature_count), name='main_input')
    dilations = [1, 2, 4, 8, 16, 32]

    tmp = Dense(math.ceil(24 * args.hidden_units), activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
    if args.dropout > 0:
        tmp = Dropout(args.dropout)(tmp)

    vad_gru = TCN(math.ceil(24 * args.hidden_units), name='vad_gru', dilations=dilations, return_sequences=True)(tmp)
    if args.dropout > 0:
        vad_gru = Dropout(args.dropout)(vad_gru)
    # vad_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="vad_gru_activation")(GRU(24, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp))
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
    noise_gru = TCN(math.ceil(48 * args.hidden_units), name='noise_gru', dilations=dilations, return_sequences=True)(noise_input)
    if args.dropout > 0:
        noise_gru = Dropout(args.dropout)(noise_gru)
    # noise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="noise_gru_activation")(GRU(48, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input))

    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
    denoise_gru = TCN(math.ceil(96 * args.hidden_units), name='denoise_gru', dilations=dilations, return_sequences=True)(denoise_input)
    if args.dropout > 0:
        denoise_gru = Dropout(args.dropout)(denoise_gru)
    # denoise_gru = keras.layers.LeakyReLU(alpha=gru_lrelu_alpha, name="denoise_gru_activation")(GRU(96, activation=None, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input))

    denoise_output = Dense(args.bands, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
elif args.arch == 'tcn':
    # 自前でresnetのcnnみたいにtcnを積み上げたもの
    main_input = Input(shape=(None, feature_count), name='main_input')
    x = main_input
    if args.input_dropout > 0:
        x = Dropout(args.input_dropout)(x)
    dilations = []
    for i in range(args.tcn_dilation_order + 1):
        dilations.append(1 << i)

    for i in range(args.tcn_layers):
        x = tcn_res_blocks(int(16 * args.hidden_units), 3, dilations, x)

    vad_output = Dense(1, activation='sigmoid', name='vad_output')(x)
    denoise_output = Dense(args.bands, activation='sigmoid', name='denoise_output')(x)
    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
else:
    raise 'unknown arch'

optimizer = keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)


if args.gpus > 0:
    compiled_model = multi_gpu_model(model, gpus=args.gpus)
else:
    compiled_model = model

compiled_model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer=optimizer, loss_weights=[10, 0.5])

print('count_params {}'.format(model.count_params()))
print(model.summary())

batch_size = args.batch_size

if args.mmap:
    print('Mmap .f32 file')
    all_data = np.memmap(args.data, dtype='float32', mode='r', shape=(os.path.getsize(args.data) // (4 * (feature_count + 2 * args.bands + 1)), feature_count + 2 * args.bands + 1));
else:
    print('Loading data from .h5...')
    all_data = np.fromfile(args.data, dtype='float32');
    all_data = np.reshape(all_data, (-1, feature_count + 2 * args.bands + 1));
    # with h5py.File(args.data, 'r') as hf:
    #     all_data = hf['data'][:]
    print('done.')

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :feature_count]
if args.noise_prob > 0:
    for i in range(nb_sequences):
        act = np.random.binomial(1, args.noise_prob, window_size // 100).repeat(100).reshape(window_size, 1)
        x_train[i * window_size:(i + 1) * window_size, :] += act * np.random.normal(0, args.noise_stddev, window_size * feature_count).reshape(window_size, feature_count)
y_train = all_data[:nb_sequences*window_size, feature_count:feature_count + args.bands]
noise_train = all_data[:nb_sequences*window_size, feature_count + args.bands:feature_count + 2 * args.bands]
vad_train = all_data[:nb_sequences*window_size, feature_count + 2 * args.bands:feature_count + 2 * args.bands + 1]

def window(ar, features):
    st = (ar.strides[0], ar.strides[0], ar.strides[1])
    return np.lib.stride_tricks.as_strided(ar, strides = st, shape = (nb_sequences*window_size - window_size + 1, window_size, features))

if args.arch == 'original' or args.arch == 'original_tcn' or args.arch == 'tcn':
    x_train = np.reshape(x_train, (nb_sequences, window_size, feature_count))
    y_train = np.reshape(y_train, (nb_sequences, window_size, args.bands))
    noise_train = np.reshape(noise_train, (nb_sequences, window_size, args.bands))
    vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))
elif args.arch == 'cnn' or args.arch == 'cnn2':
    if args.window_overlap > 0:
        x_train = window(x_train, feature_count)[::args.window_overlap,:,:]
        y_train = window(y_train, args.bands)[::args.window_overlap,window_size - 1,:]
        vad_train = window(vad_train, 1)[::args.window_overlap,window_size - 1,:]
    else:
        x_train = np.reshape(x_train, (nb_sequences, window_size, feature_count))
        y_train = np.reshape(y_train, (nb_sequences, window_size, args.bands))[:,window_size - 1,:]
        vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))[:,window_size - 1,:]
else:
    raise 'unknown arch'

all_data = 0
print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

print('Train...')
dir = datetime.datetime.now().strftime("train%Y%m%d_%H%M%S")
os.makedirs(os.path.join(dir), exist_ok=True)

# plot_model(model, to_file=dir + "/model.png")

# class DumpToSimpleCppCallback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         with open(dir + '/cppmodel{}.nnet'.format(epoch), 'w') as fout:
#             dump_to_simple_cpp.dump_to_simple_cpp(model, fout)
#
# class DumpPocketTensor(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         pt.export_model(model, dir + '/cppmodel{}.nnet'.format(epoch))

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath = os.path.join(dir, 'weights.{epoch:03d}-{val_loss:.2f}.hdf5'),
                                                  monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='min',
                                                  period=1)

if args.mixup == 0:
    compiled_model.fit(x_train, [y_train, vad_train],
              batch_size=batch_size,
              epochs=120,
              validation_split=0.1,
              callbacks=[modelCheckpoint])
else:
    x_train_train, x_val, y_train_train, y_val, vad_train_train, vad_val = train_test_split(x_train, y_train, vad_train, test_size=0.1, shuffle=True, random_state=1)

    # train_gen = MySequence(x_train_train, y_train_train, vad_train_train, batch_size)
    train_gen = MyMixupGenerator(x_train_train, y_train_train, vad_train_train, batch_size, args.mixup_alpha)
    # train_gen = mixup_generator.MixupGenerator(x_train_train, np.array([y_train_train, vad_val]), batch_size=batch_size, alpha=0.2)()
    val_gen = MySequence(x_val, y_val, vad_val, batch_size)
    compiled_model.fit_generator(
        generator=train_gen(),
        epochs=120,
        steps_per_epoch=args.mixup * len(train_gen),
        use_multiprocessing=True,
        workers=4,
        verbose=1,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=[modelCheckpoint])

model.save("newweights9i.hdf5")


