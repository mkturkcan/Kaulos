from compact_dependencies import *
from keras.layers.recurrent import *
import sys
sys.setrecursionlimit(10000)
from keras.layers import Lambda
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Lambda, Concatenate, Input, Reshape, BatchNormalization
from keras.layers import SimpleRNN, GRU, LSTM
from keras import initializers
from keras.optimizers import RMSprop, Adam
from keras.initializers import Constant
from keras.models import Model
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

_BACKEND = 'theano'
class LeakyIAF(Layer):
    def __init__(self, threshold, alpha, **kwargs):
        self.threshold = threshold
        self.alpha = alpha
        super(LeakyIAF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_shape_m = input_shape
        print(input_shape[1:])
        self.V = self.add_weight(
            shape=(1,1),
            name='V',
            initializer='zeros',
            trainable=False)
        super(LeakyIAF, self).build(input_shape)
    def call(self, x):
        V = self.V + x
        if (_BACKEND == 'tensorflow'):
            import tensorflow as tf
            V = tf.where(K.greater(V, self.threshold), 0. * V, V)
        else:
            V = K.switch(K.greater(V, self.threshold), 0. * V, V)
        updates = []
        updates.append((self.V, V))
        self.add_update(updates)
        return self.V
    def compute_output_shape(self, input_shape):
        return input_shape


class HodgkinHuxley(Layer):
    def __init__(self, dt, threshold, alpha, **kwargs):
        self.dt = dt
        self.threshold = threshold
        self.alpha = alpha
        super(HodgkinHuxley, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_shape_m = input_shape
        print(input_shape[1:])
        self.V = self.add_weight(
            shape=(1,1),
            name='V',
            initializer='zeros',
            trainable=False)
        self.n = self.add_weight(
            shape=(1,1),
            name='n',
            initializer='zeros',
            trainable=False)
        self.m = self.add_weight(
            shape=(1,1),
            name='m',
            initializer='zeros',
            trainable=False)
        self.h = self.add_weight(
            shape=(1,1),
            name='h',
            initializer='ones',
            trainable=False)
        self.g_K = 36
        self.g_Na = 120
        self.g_R = 0.3
        self.E = [-12, 115, 10.613]
        super(HodgkinHuxley, self).build(input_shape)
    def call(self, I_ext):
        a_1 = (10-self.V)/(100*(K.exp((10-self.V)/10)-1))
        a_2 = (25-self.V)/(10*(K.exp((25-self.V)/10)-1))
        a_3 = 0.07*K.exp(-self.V/20)

        b_1 = 0.125*K.exp(-self.V/80)
        b_2 = 4*K.exp(-self.V/18)
        b_3 = 1/(K.exp((30-self.V/10)+1))

        m = self.m + self.dt*(a_1*(1-self.m) - b_1*self.m)
        n = self.n + self.dt*(a_2*(1-self.n) - b_2*self.n)
        h = self.h + self.dt*(a_3*(1-self.h) - b_3*self.h)

        gnmh_1 = self.g_K*K.pow(m, 4)
        gnmh_2 = self.g_Na*K.pow(n, 3)*h
        gnmh_3 = self.g_R
        I = (gnmh_1 *(self.V-self.E[0])) + (gnmh_2 *(self.V-self.E[1])) + (gnmh_3 *(self.V-self.E[2]))
        V = self.V + self.dt*(I_ext - I)
        updates = []
        updates.append((self.m, m))
        updates.append((self.n, n))
        updates.append((self.h, h))
        updates.append((self.V, V))
        self.add_update(updates)
        return self.V
    def compute_output_shape(self, input_shape):
        return input_shape



class _Relay():
    def __init__(self, **kwargs):
		self.params = {}
		self.updates = {}
		self.accesses = []
		self.states = {}

		options.update(kwargs)
