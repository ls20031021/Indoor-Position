#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This model is to tran an independent model for IMU dataset
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
#epoch = FLAGS.epoch
#learning_rate = FLAGS.learning_rate
epoch = 300
learning_rate = 0.0025
scenario=FLAGS.scenario
model_name = 'IMU_lstm_Model'



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


IMUTrain=SensorTrain[:,:,0:2]
IMUVal=SensorVal[:,:,0:2]
IMUTest=SensorTest[:,:,0:2]

imuinput=Input(shape=(IMUTrain.shape[1], IMUTrain.shape[2]))
imuoutput=LSTM(input_shape=(IMUTrain.shape[1], IMUTrain.shape[2]),units=hidden_size,name="IMU_Feature")(imuinput)
output=Dense(2,activation='relu')(imuoutput)
imu_mmloc=Model(inputs=[imuinput],outputs=[output])
imu_mmloc.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=['acc'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

imu_mmloc.fit([IMUTrain], locationtrain,
                       validation_data=([IMUVal],locationval),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

#save model
imu_mmloc.save(scenario+"/model/"+str(model_name)+".h5")


#Plot trajectory
locPrediction = imu_mmloc.predict([SensorTest,WifiTest], batch_size=batch_size)
aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
#visualization for error line and location prediction
v.visualization(locationtest,locPrediction,model_name)
#print location prediction picture
v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
#draw cdf picture
v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)