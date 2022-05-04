# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:20:45 2020

@author: mwei_archor
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
from data_functions import combine_overlap_wifi,normalisation,overlapping,read_overlap_data,downsample_data,DownsampleDataset,WifinozerosDataset,SensorDataset,SimpleDownsampling
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard

time_step=100

'''
sensor=SensorDataset()
downsample=DownsampleDataset()
wifinonzero=WifinozerosDataset()

SensorTest=sensor.testx
locationtest=sensor.testy

DownsampleSensorTest=downsample.sensortest
Downsamplelocationtest=downsample.labeltest
DownsampleWifiTest=downsample.wifitest

wifinonzeroTest=wifinonzero.testx
wifinonzerolocationtest=wifinonzero.testy
'''
SensorTrain14, location14 = overlapping('overlap_timestep100/14_timestep100.csv',3, time_step)
DownsampleSensorTest=SimpleDownsampling(SensorTrain14,10)

DownsampleWifiTest,Downsamplelocationtest=combine_overlap_wifi('sensordata/wifi_overlap_14_timestp100.csv')


#sensor_baseline=load_model("model/sensor_baseline_model.h5")
sensor_downsample=load_model("msmodel/overlap_downsample_sensor_ms.h5")
wifi_all=load_model("msmodel/wifi_ms.h5")
#wifi_nonzero=load_model("model/wifi_DNN_nonzero_model.h5")
mmloc_multi=load_model("msmodel/mmloc_ms_overlap.h5")

#sensorbaselineloc = sensor_baseline.predict(SensorTest, batch_size=100)
sensordownsampleloc = sensor_downsample.predict(DownsampleSensorTest, batch_size=100)
wifiloc = wifi_all.predict(DownsampleWifiTest, batch_size=100)
#wifinonzeroloc = wifi_nonzero.predict(wifinonzeroTest, batch_size=100)
mmloc_multiloc = mmloc_multi.predict([DownsampleSensorTest,DownsampleWifiTest], batch_size=100)

bin_edgeses=[]
cdfs=[]
names=["sensor_downsample","wifi","mmloc_overlap"]
colors=['green','blue','gray']

#bin_edge,cdf=pf.cdfdiff(target=locationtest, predict=sensorbaselineloc)
#bin_edgeses.append(bin_edge)
#cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=sensordownsampleloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=wifiloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
#bin_edge,cdf=pf.cdfdiff(target=wifinonzerolocationtest, predict=wifinonzeroloc)
#bin_edgeses.append(bin_edge)
#cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_multiloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)


fig = plt.figure()
for i in range(len(cdfs)): 
    plt.plot(bin_edgeses[i][0:-1], cdfs[i],linestyle='--', label=names[i],color=colors[i-1])
    
plt.xlim(xmin = 0)
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.legend(names,loc='upper right')
plt.grid(True)
plt.title('Testing CDF')
fig.savefig("CDF.pdf")

