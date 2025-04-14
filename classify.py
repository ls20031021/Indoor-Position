#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


def get_imu_labels(df, pointsnumber, step):
    count=0
    label=0
    round=0
    num=0
    a=np.zeros((1,2))
    #for num in range(0,2*(pointsnumber-2)+1):
    for i in df.iloc[:,0]:
        if  count==0:
            t=np.array([[i-1,label]])
            a=np.concatenate((a,t),axis=0)
            count=count+1
            label=label+1
            #num=num+1
            # if num==df.iloc[i,0]-step:
            #     label=label+1
            #     print('here1')
            
    
        elif 0 < count < pointsnumber-1:
            t=np.array([[i-step,label]])
            a=np.concatenate((a,t),axis=0)
            label=label+1
            t=np.array([[i+step,label]])
            a=np.concatenate((a,t),axis=0)
            count=count+1
            label=label+1
            if label==(2*(pointsnumber-2)+1):
                label=2*(pointsnumber-2)
    
        else:
            t=np.array([[i,label]])
            a=np.concatenate((a,t),axis=0)
            count=0
            label=0
    
            round=round+1
            print('Round '+str(round)+' finish')
    a=a[1::] #drop the first initial row of zeros
    
    cnt=0
    b=np.zeros((1,2))
    
    for i in range (0, int(a[-1,0])+1):
        if i == a[cnt,0]:
            tag=a[cnt,1]
            cnt=cnt+1
        tem=np.array([[i,tag]])
        b=np.concatenate((b,tem),axis=0)
    b=b[1::]    
    return b
