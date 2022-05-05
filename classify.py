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

trainA=pd.read_csv('points/trainA.csv', header=None)
valA=pd.read_csv('points/valA.csv', header=None)
testA=pd.read_csv('points/testA.csv', header=None)

trainB=pd.read_csv('points/trainB.csv', header=None)
valB=pd.read_csv('points/valB.csv', header=None)
testB=pd.read_csv('points/testB.csv', header=None)

scenario='scenarioA'

SensorTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_train.npy")
locationtrainA=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_train.npy")

SensorVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_val.npy")
locationvalA=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_val.npy")

SensorTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_test.npy")
locationtestA=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")


plt.scatter(locationtestA[:,0],locationtestA[:,1])
plt.scatter(locationtestA[(841-20):(841+20),0],locationtestA[(841-20):(841+20),1])




totalpoints=5

for i in range(len(testA)//totalpoints):
    for j in range (0,totalpoints):
        if 1 < j < totalpoints:
            a
        print (testA.iloc[j,0])
    print ('round'+str(i))

for j in range(len(df)):    
    for q in range (0, df.iloc[-1,0]):
        a=np.append(a,label)       
        if q == df.iloc[j,0]:
            label=label+1
            print ('break when q is '+str(q))
        break
    print(j)
    
df=testA

a=[]
q=0    

label=0
    
a=[]  
for q in range (0, df.iloc[-1,0]+1):
    a=np.append(a,label) 
    if q == df.iloc[label,0]:
        label=label+1
        print ('break when q is '+str(q))
 
a=a[1::]
