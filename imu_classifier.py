#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:13:21 2022
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
from tensorflow.keras.layers import Dense, concatenate, LSTM,Input,ReLU,Multiply,Add,GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.keras.utils import plot_model

#Choose Scenario: Type A or B to select loading data.
FLAGS=v.choose_scenario('A')

wifi_input_size = FLAGS.wifi_input_size
hidden_size = FLAGS.hidden_size
batch_size = FLAGS.batch_size
scenario=FLAGS.scenario

epoch = 100
learning_rate = 0.01
pointsnumber=5
step=20
model_name = 'imu_classiferA'


#Load data
SensorTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_train.npy")
locationtrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_train.npy")
WifiTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_train.npy")

SensorVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_val.npy")
locationval=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_val.npy")
WifiVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_val.npy")

SensorTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_test.npy")
locationtest=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")
WifiTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_test.npy")

####IMU Data
IMUTrain=SensorTrain[:,:,0:2]
IMUVal=SensorVal[:,:,0:2]
IMUTest=SensorTest[:,:,0:2]

####IMU Data
MagTrain=SensorTrain[:,:,2:3]
MagVal=SensorVal[:,:,2:3]
MagTest=SensorTest[:,:,2:3]

####IMU class labels
trainA=pd.read_csv('points/trainA.csv', header=None)
valA=pd.read_csv('points/valA.csv', header=None)
testA=pd.read_csv('points/testA.csv', header=None)

trainB=pd.read_csv('points/trainB.csv', header=None)
valB=pd.read_csv('points/valB.csv', header=None)
testB=pd.read_csv('points/testB.csv', header=None)



imu_train_class = v.get_imu_labels(trainA, pointsnumber, step)
tags = np.unique(imu_train_class)
num_class=tags.shape[0]

imu_val_class = v.get_imu_labels(valA, pointsnumber, step)

imu_test_class = v.get_imu_labels(testA, pointsnumber, step)

IMUTrain=SensorTrain[:,:,0:2]
IMUVal=SensorVal[:,:,0:2]
IMUTest=SensorTest[:,:,0:2]



imuinput=Input(shape=(IMUTrain.shape[1], IMUTrain.shape[2]))
imuoutput=LSTM(input_shape=(IMUTrain.shape[1], IMUTrain.shape[2]),units=hidden_size,name="IMU_Feature")(imuinput)
output=Dense(num_class,activation='softmax')(imuoutput)

imu_classfier=Model(inputs=[imuinput],outputs=[output])
imu_classfier.compile(
    optimizer=Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
imu_classfier.fit([IMUTrain], imu_train_class,
                       validation_data=([IMUVal],imu_val_class),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )


