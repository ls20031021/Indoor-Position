#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:14:19 2022

@author: weixijia
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow.compat.v1 as tf

import visualization as v
import pandas as pd
from absl import flags
from tensorflow.compat.v1 import flags

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, concatenate, LSTM,Input,ReLU,Multiply,Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.keras.utils import plot_model

trainA=pd.read_csv('points/trainA.csv')
valA=pd.read_csv('points/valA.csv')
testA=pd.read_csv('points/testA.csv')

trainB=pd.read_csv('points/trainB.csv')
valB=pd.read_csv('points/valB.csv')
testB=pd.read_csv('points/testB.csv')



locationtrainA=np.load("scenarioA/overlap_timestep1000/overlap_ds_location_train.npy")


locationvalA=np.load("scenarioA/overlap_timestep1000/overlap_ds_location_val.npy")


locationtestA=np.load("scenarioA/overlap_timestep1000/overlap_ds_location_test.npy")
