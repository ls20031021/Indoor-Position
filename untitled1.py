#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:44:50 2022

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

testA=pd.read_csv('points/testA.csv', header=None)

scenario='scenarioA'

SensorTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_test.npy")
locationtestA=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")

trainA=pd.read_csv('points/trainA.csv', header=None)
valA=pd.read_csv('points/valA.csv', header=None)
testA=pd.read_csv('points/testA.csv', header=None)

trainB=pd.read_csv('points/trainB.csv', header=None)
valB=pd.read_csv('points/valB.csv', header=None)
testB=pd.read_csv('points/testB.csv', header=None)


df=trainA


pointsnumber=5
step=20
count=0
round=0
a=[[],[]]
for i in df.iloc[:,0]:
    if  count==0:
        a=np.append(a,i-1)

        count=count+1

    elif 0 < count < pointsnumber-1:
        a=np.append(a,i-step)
        a=np.append(a,i+step)

        count=count+1
    else:
        a=np.append(a,i)

        count=0
        round=round+1
        print('Round '+str(round)+' finish')