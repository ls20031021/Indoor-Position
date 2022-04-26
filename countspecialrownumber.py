#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:06:32 2022

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

d1=pd.read_csv("scenarioB/csv/1_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d2=pd.read_csv("scenarioB/csv/2_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d3=pd.read_csv("scenarioB/csv/3_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d4=pd.read_csv("scenarioB/csv/4_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d5=pd.read_csv("scenarioB/csv/5_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d6=pd.read_csv("scenarioB/csv/6_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d7=pd.read_csv("scenarioB/csv/7_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d8=pd.read_csv("scenarioB/csv/8_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d9=pd.read_csv("scenarioB/csv/9_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d10=pd.read_csv("scenarioB/csv/10_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d11=pd.read_csv("scenarioB/csv/11_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d12=pd.read_csv("scenarioB/csv/12_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d13=pd.read_csv("scenarioB/csv/13_timestep1000_overlap900.csv",usecols=['st','lat','lng'])
d14=pd.read_csv("scenarioB/csv/14_timestep1000_overlap900.csv",usecols=['st','lat','lng'])

train=pd.concat([d1, d2, d3, d4, d5, d6, d7, d8])
train=train.reset_index()

#scenarioA
# turing_points_row=train.index[
#     ((train.lat == 33.19) & (train.lng== 32.63)) | 
#     ((train.lat == 0) & (train.lng== 23.76)) | 
#     ((train.lat == 6.36) & (train.lng== 0))   | 
#     ((train.lat == 39.76) & (train.lng== 8.92))
#     ]

#scenarioB
turing_points_row=train.index[
    ((train.lat == 22.0) & (train.lng== 12.85)) | 
    ((train.lat == 16.7) & (train.lng== 12.85)) | 
    ((train.lat == 16.7) & (train.lng== 21.55)) | 
    ((train.lat == 22.0) & (train.lng== 21.55)) |
    ((train.lat == 22.0) & (train.lng== 20.2)) | 
    ((train.lat == 23.9) & (train.lng== 20.2)) | 
    ((train.lat == 23.9) & (train.lng== 17.0)) | 
    ((train.lat == 23.9) & (train.lng== 17.0)) | 
    ((train.lat == 22.0) & (train.lng== 17.0)) 
    ]

df = pd.DataFrame(data=turing_points_row)
df.to_csv('trainrow.csv')

val=pd.concat([d9,d10,d11,d12,d13])
val=val.reset_index()

#scenarioA
# val_points_row=val.index[
#     ((val.lat == 33.19) & (val.lng== 32.63)) | 
#     ((val.lat == 0) & (val.lng== 23.76)) | 
#     ((val.lat == 6.36) & (val.lng== 0))   | 
#     ((val.lat == 39.76) & (val.lng== 8.92))
#     ]

#scenarioB
val_points_row=val.index[
    ((val.lat == 22.0) & (val.lng== 12.85)) | 
    ((val.lat == 16.7) & (val.lng== 12.85)) | 
    ((val.lat == 16.7) & (val.lng== 21.55)) | 
    ((val.lat == 22.0) & (val.lng== 21.55)) |
    ((val.lat == 22.0) & (val.lng== 20.2)) | 
    ((val.lat == 23.9) & (val.lng== 20.2)) | 
    ((val.lat == 23.9) & (val.lng== 17.0)) | 
    ((val.lat == 23.9) & (val.lng== 17.0)) | 
    ((val.lat == 22.0) & (val.lng== 17.0)) 
    ]


df = pd.DataFrame(data=val_points_row)
df.to_csv('valrow.csv')