#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:05:50 2022

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
FLAGS=v.choose_scenario('B')

wifi_input_size = FLAGS.wifi_input_size
hidden_size = FLAGS.hidden_size
batch_size = FLAGS.batch_size
scenario=FLAGS.scenario

epoch = 1
learning_rate = 0.0025
model_name = 'MTLocBforplot'

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


#Load trained IMU Model
IMU_Model=load_model("scenarioA/model/IMU_Model.h5")

# for i in range (0,len(model.layers)):
#     print (i, model.layers[i])

#extract lstm layer from trained model    
lstm_extracted=IMU_Model.get_layer('lstm')
lstm_extracted.trainable = False #make transfered layer non-trainable


#Construct mmloc model


imuinput=Input(shape=(IMUTrain.shape[1], IMUTrain.shape[2]),name="IMU-LSTM")
imuoutput=lstm_extracted(imuinput)


maginput=Input(shape=(MagTrain.shape[1], MagTrain.shape[2]),name="MAG-LSTM")
magoutput=LSTM(input_shape=(MagTrain.shape[1], MagTrain.shape[2]),units=hidden_size,name="MAG_Feature")(maginput)

wifiinput=Input(shape=(wifi_input_size,),name="WiFi-DNN")
wifi=Dense(hidden_size,name="Dense_WiFi_1")(wifiinput)
wifi=ReLU()(wifi)
wifi=Dense(hidden_size,name="Dense_WiFi_2")(wifi)
wifi=ReLU()(wifi)
wifioutput=Dense(hidden_size,name="WiFi_Feature")(wifi)

merge=concatenate([imuoutput,magoutput,wifioutput],name="Fusion")
hidden=Dense(hidden_size,activation='relu',name="Dense_Fusion")(merge)
output=Dense(2,activation='relu',name="Location")(hidden)
MTLoc=Model(inputs=[imuinput,maginput,wifiinput],outputs=[output])

MTLoc.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=['acc'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

MTLoc.fit([IMUTrain,MagTrain,WifiTrain], locationtrain,
                       validation_data=([IMUVal,MagVal,WifiVal],locationval),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

#save model
MTLoc.save(scenario+"/model/"+str(model_name)+"plot.h5")
