#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:49:09 2022

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
MTLocB = 'MTLocB'
MTLocB=load_model(scenario+"/model/"+str(MTLocB)+".h5")
locPredictionMTLocB = MTLocB.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB,cdfMTLocB=v.cdfdiff(target=locationval,predict=locPredictionMTLocB)
#aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)

MTLocB_8plot = 'MTLocB_8plot'
MTLocB_8plot=load_model(scenario+"/model/"+str(MTLocB_8plot)+".h5")
locPredictionMTLocB_8plot = MTLocB_8plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_8plot,cdfMTLocB_8plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_8plot)

MTLocB_7plot = 'MTLocB_7plot'
MTLocB_7plot=load_model(scenario+"/model/"+str(MTLocB_7plot)+".h5")
locPredictionMMTLocB_7plot = MTLocB_7plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_7plot,cdfMTLocB_7plot=v.cdfdiff(target=locationval,predict=locPredictionMMTLocB_7plot)

MTLocB_6plot = 'MTLocB_6plot'
MTLocB_6plot=load_model(scenario+"/model/"+str(MTLocB_6plot)+".h5")
locPredictionMTLocB_6plot = MTLocB_6plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_6plot,cdfMTLocB_6plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_6plot)

MTLocB_5plot = 'MTLocB_5plot'
MTLocB_5plot=load_model(scenario+"/model/"+str(MTLocB_5plot)+".h5")
locPredictionMTLocB_5plot = MTLocB_5plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_5plot,cdfMTLocB_5plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_5plot)

MTLocB_4plot = 'MTLocB_4plot'
MTLocB_4plot=load_model(scenario+"/model/"+str(MTLocB_4plot)+".h5")
locPredictionMTLocB_4plot = MTLocB_4plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_4plot,cdfMTLocB_4plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_4plot)

MTLocB_3plot = 'MTLocB_3plot'
MTLocB_3plot=load_model(scenario+"/model/"+str(MTLocB_3plot)+".h5")
locPredictionMTLocB_3plot = MTLocB_3plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_3plot,cdfMTLocB_3plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_3plot)

MTLocB_2plot = 'MTLocB_2plot'
MTLocB_2plot=load_model(scenario+"/model/"+str(MTLocB_2plot)+".h5")
locPredictionMTLocB_2plot = MTLocB_2plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_2plot,cdfMTLocB_2plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_2plot)

MTLocB_1plot = 'MTLocB_1plot'
MTLocB_1plot=load_model(scenario+"/model/"+str(MTLocB_1plot)+".h5")
locPredictionMTLocB_1plot = MTLocB_1plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_1plot,cdfMTLocB_1plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_1plot)

MTLocB_0plot = 'MTLocB_0plot'
MTLocB_0plot=load_model(scenario+"/model/"+str(MTLocB_0plot)+".h5")
locPredictionMTLocB_0plot = MTLocB_0plot.predict([IMUVal,MagVal,WifiVal], batch_size=batch_size)
bin_edgeMTLocB_0plot,cdfMTLocB_0plot=v.cdfdiff(target=locationval,predict=locPredictionMTLocB_0plot)





fig=plt.figure()

plt.plot(bin_edgeMTLocB[0:-1],cdfMTLocB,linestyle='-',label=str('train from scratch'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_8plot[0:-1],cdfMTLocB_8plot,linestyle='-',label=str('100%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_7plot[0:-1],cdfMTLocB_7plot,linestyle='-',label=str('87.5%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_6plot[0:-1],cdfMTLocB_6plot,linestyle='-',label=str('75.0%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_5plot[0:-1],cdfMTLocB_5plot,linestyle='-',label=str('62.5%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_4plot[0:-1],cdfMTLocB_4plot,linestyle='-',label=str('50.0%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_3plot[0:-1],cdfMTLocB_3plot,linestyle='-',label=str('37.5%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_2plot[0:-1],cdfMTLocB_2plot,linestyle='-',label=str('25.0%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_1plot[0:-1],cdfMTLocB_1plot,linestyle='-',label=str('12.5%'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_0plot[0:-1],cdfMTLocB_0plot,linestyle='-',label=str('0.0%'),linewidth=0.7)



plt.xlim([0, 20])
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.grid(True)
plt.title(('Fine Tuning Validation CDF'))
plt.legend(loc='lower right')
fig.savefig("Fine-tuning_CDF(val).pdf")
print('cdf plotted')


# fig=plt.figure()

# plt.plot(bin_edgeMTLocB[0:-1],cdfMTLocB,linestyle='-',label=str('train from scratch'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_8plot[0:-1],cdfMTLocB_8plot,linestyle='-',label=str('100%'),linewidth=0.7)
# #plt.plot(bin_edgeMTLocB_7plot[0:-1],cdfMTLocB_7plot,linestyle='-',label=str('87.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_6plot[0:-1],cdfMTLocB_6plot,linestyle='-',label=str('75.0%'),linewidth=0.7)
# #plt.plot(bin_edgeMTLocB_5plot[0:-1],cdfMTLocB_5plot,linestyle='-',label=str('62.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_4plot[0:-1],cdfMTLocB_4plot,linestyle='-',label=str('50.0%'),linewidth=0.7)
# #plt.plot(bin_edgeMTLocB_3plot[0:-1],cdfMTLocB_3plot,linestyle='-',label=str('37.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_2plot[0:-1],cdfMTLocB_2plot,linestyle='-',label=str('25.0%'),linewidth=0.7)
# #plt.plot(bin_edgeMTLocB_1plot[0:-1],cdfMTLocB_1plot,linestyle='-',label=str('12.5%'),linewidth=0.7)
# plt.plot(bin_edgeMTLocB_0plot[0:-1],cdfMTLocB_0plot,linestyle='-',label=str('0.0%'),linewidth=0.7)



# plt.xlim([0, 20])
# plt.ylim((0,1))
# plt.xlabel("metres")
# plt.ylabel("CDF")
# plt.grid(True)
# plt.title(('Fine Tuning CDF (5 level)'))
# plt.legend(loc='lower right')
# fig.savefig("Fine-tuning_CDF(5level).pdf")
# print('cdf plotted')