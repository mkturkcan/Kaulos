from __future__ import division, print_function
import gc
import json
import h5py
import pickle
import random
import glob, os
import numpy as np
import pandas as pd
from keras import backend as K
from scipy.stats import zscore
from keras.layers import Input
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.local import LocallyConnected2D
from sklearn.decomposition import IncrementalPCA, PCA
from keras.layers.advanced_activations import PReLU, ELU
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler, normalize
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from sklearn.cv import StratifiedKFold, LeavePOut, LeaveOneOut
from keras.layers import Dense, Input, Lambda, Dropout, Activation, Flatten, Dropout
from keras.layers import Convolution3D, Flatten, MaxPooling3D, UpSampling3D, Reshape, ZeroPadding3D
from keras.layers import Convolution2D, Flatten, MaxPooling2D, UpSampling2D, Reshape, ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers import Convolution1D, Flatten, MaxPooling1D, UpSampling1D, Reshape, ZeroPadding1D, AveragePooling1D, GlobalMaxPooling1D

np.random.seed(1337)

def save_pickle(file_name, variable):
    output = open(file_name, 'wb')
    pickle.dump(variable, output)
    output.close()

def load_pickle(file_name):
    pkl_file = open(file_name, 'rb')
    variable = pickle.load(pkl_file)
    pkl_file.close()
    return variable

def save_large_dataset(file_name, variable):
    h5f = h5py.File(file_name + '.h5', 'w')
    h5f.create_dataset('variable', data=variable)
    h5f.close()

def load_large_dataset(file_name):
    h5f = h5py.File(file_name + '.h5','r')
    variable = h5f['variable'][:]
    h5f.close()
    return variable
