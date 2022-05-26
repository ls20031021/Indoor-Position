#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:22:39 2022

@author: weixijia
"""


import numpy as np
import matplotlib.pyplot as plt
import math
import random
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
original_imu_test=IMUTest

####IMU Data
MagTrain=SensorTrain[:,:,2:3]
MagVal=SensorVal[:,:,2:3]
MagTest=SensorTest[:,:,2:3]
original_mag_test=IMUTest


wifi_ratio=0.4
mag_ratio=0
imu_ratio=0

mag_random_list = []
for i in range(0, int(len(MagTest)*mag_ratio)-1):
    mag_random_list.append(random.randint(0, len(MagTest)))

if 0 < mag_ratio < 1:
    for i in mag_random_list:
        MagTest[i-1,:,:]=0
    print (str(mag_ratio*100)+'% of the Mag input masked')
elif mag_ratio==1:
    MagTest[:,:,:]=0
    
imu_random_list = []
for i in range(0, int(len(IMUTest)*imu_ratio)-1):
    imu_random_list.append(random.randint(0, len(IMUTest)))

if 0 < imu_ratio < 1:
    for i in imu_random_list:
        IMUTest[i-1,:,:]=0
    print (str(imu_ratio*100)+'% of the IMU input masked')
elif imu_ratio==1:
    IMUTest[:,:,:]=0

wifi_random_list = []
for i in range(0, int(len(WifiTest)*wifi_ratio)-1):
    wifi_random_list.append(random.randint(0, len(WifiTest)))

if 0 < wifi_ratio < 1:
    for i in wifi_random_list:
        WifiTest[i-1,:]=0
    print (str(wifi_ratio*100)+'% of the WiFi input masked')
elif wifi_ratio==1:
    WifiTest[:,:]=0
    
SensorTest=np.concatenate((IMUTest, MagTest),axis=2)
    

#Load Model
MTLocB = 'MMLocB'
MTLocB=load_model(scenario+"/model/"+str(MTLocB)+".h5")

locPredictionMTLocB = MTLocB.predict([SensorTest,WifiTest], batch_size=batch_size)
bin_edgeMTLocB,cdfMTLocB=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB)
#aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)




MTLocB_5plot = 'MTLocB_5plot'
MTLocB_5plot=load_model(scenario+"/model/"+str(MTLocB_5plot)+".h5")
locPredictionMTLocB_5plot = MTLocB_5plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
bin_edgeMTLocB_5plot,cdfMTLocB_5plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_5plot)

# MTLocB_4plot = 'MTLocB_4plot'
# MTLocB_4plot=load_model(scenario+"/model/"+str(MTLocB_4plot)+".h5")
# locPredictionMTLocB_4plot = MTLocB_4plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
# bin_edgeMTLocB_4plot,cdfMTLocB_4plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_4plot)

# MTLocB_3plot = 'MTLocB_3plot'
# MTLocB_3plot=load_model(scenario+"/model/"+str(MTLocB_3plot)+".h5")
# locPredictionMTLocB_3plot = MTLocB_3plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
# bin_edgeMTLocB_3plot,cdfMTLocB_3plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_3plot)

# MTLocB_2plot = 'MTLocB_2plot'
# MTLocB_2plot=load_model(scenario+"/model/"+str(MTLocB_2plot)+".h5")
# locPredictionMTLocB_2plot = MTLocB_2plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
# bin_edgeMTLocB_2plot,cdfMTLocB_2plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_2plot)

# MTLocB_1plot = 'MTLocB_1plot'
# MTLocB_1plot=load_model(scenario+"/model/"+str(MTLocB_1plot)+".h5")
# locPredictionMTLocB_1plot = MTLocB_1plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
# bin_edgeMTLocB_1plot,cdfMTLocB_1plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_1plot)

MTLocB_0plot = 'MTLocB_0plot'
MTLocB_0plot=load_model(scenario+"/model/"+str(MTLocB_0plot)+".h5")
locPredictionMTLocB_0plot = MTLocB_0plot.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
bin_edgeMTLocB_0plot,cdfMTLocB_0plot=v.cdfdiff(target=locationtest,predict=locPredictionMTLocB_0plot)





fig=plt.figure()

plt.plot(bin_edgeMTLocB[0:-1],cdfMTLocB,linestyle='-',label=str('Raw model'),linewidth=1)
# plt.plot(bin_edgeMTLocB_8plot[0:-1],cdfMTLocB_8plot,linestyle='-',label=str('100%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_7plot[0:-1],cdfMTLocB_7plot,linestyle='-',label=str('87.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_6plot[0:-1],cdfMTLocB_6plot,linestyle='-',label=str('75.0%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_5plot[0:-1],cdfMTLocB_5plot,linestyle='-',label=str('Fine-tuned model'),linewidth=1)
# plt.plot(bin_edgeMTLocB_4plot[0:-1],cdfMTLocB_4plot,linestyle='-',label=str('50.0%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_3plot[0:-1],cdfMTLocB_3plot,linestyle='-',label=str('37.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_2plot[0:-1],cdfMTLocB_2plot,linestyle='-',label=str('25.0%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_1plot[0:-1],cdfMTLocB_1plot,linestyle='-',label=str('12.5%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_0plot[0:-1],cdfMTLocB_0plot,linestyle='-',label=str('Transferred model'),linewidth=1)



plt.xlim([0, 20])
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.grid(True)
plt.title(('RSS Missing CDF'))
plt.legend(loc='lower right')
fig.savefig("rss_missing.pdf")
print('cdf plotted')
