#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:27:09 2022

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



#1: IMU Independent
#2: WiFi Independent
#3: MMLocA
#4: MMLocB

mode=4

if mode==1:
    model_name = 'IMU_Model'
    #Choose Scenario: Type A or B to select loading data.
    FLAGS=v.choose_scenario('A')

    wifi_input_size = FLAGS.wifi_input_size
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    scenario=FLAGS.scenario


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
    

    #Load Model
    model=load_model(str(scenario)+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict(IMUTest, batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    #visualization for error line and location prediction
    v.visualization(locationtest,locPrediction,model_name)
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
    
if mode==2:
    print('have not defined yet')
    
if mode==3:
    model_name = 'MTLocA'
    
    #Choose Scenario: Type A or B to select loading data.
    FLAGS=v.choose_scenario('A')

    wifi_input_size = FLAGS.wifi_input_size
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    scenario=FLAGS.scenario


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
    
    #Load Model
    model=load_model(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    #visualization for error line and location prediction
    v.visualization(locationtest,locPrediction,model_name)
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)

if mode==4:
    model_name = 'MTLocB'
    #Choose Scenario: Type A or B to select loading data.
    FLAGS=v.choose_scenario('B')

    wifi_input_size = FLAGS.wifi_input_size
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    scenario=FLAGS.scenario


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
    #Load Model
    model=load_model(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    #visualization for error line and location prediction
    v.visualization(locationtest,locPrediction,model_name)
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
