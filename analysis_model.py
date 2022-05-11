#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 19:44:03 2022

@author: weixijia
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
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


model_name = 'MTLocB_0plot'
#Choose Scenario: Type A or B to select loading data.
FLAGS=v.choose_scenario('B')

wifi_input_size = FLAGS.wifi_input_size
hidden_size = FLAGS.hidden_size
batch_size = FLAGS.batch_size
scenario=FLAGS.scenario


#Load Model
model=load_model(scenario+"/model/"+str(model_name)+".h5")

def Analysis(model,model_name,scenario,batch_size):
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

    locPrediction = model.predict([IMUTest,MagTest,WifiTest], batch_size=batch_size)
    bin_edge,cdf=v.cdfdiff(target=locationtest,predict=locPrediction)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    
    #Plot path with error
    data=v.normalized_data_to_utm(np.hstack((locationtest, locPrediction)))
    datare1=data[:,0].reshape(-1,1)
    datare2=data[:,1].reshape(-1,1)
    datare3=data[:,2].reshape(-1,1)
    datare4=data[:,3].reshape(-1,1)
    Y_test=np.hstack((datare1,datare2))
    Y_pre=np.hstack((datare3,datare4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True,linestyle='--',linewidth=1)
    ax.set_xlabel("X-longitude")
    ax.set_ylabel("Y-latitude")
    Y_test = np.array(Y_test)
    for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
        if math.sqrt((pred[0]-target[0])**2 + (pred[1]-target[1])**2) < 10:
            plt.plot([pred[0], target[0]], [pred[1], target[1]], color='r',
                     linewidth=0.5, label='Error Line' if i == 0 else "")
            plt.scatter(pred[0], pred[1], label='Prediction' if i == 0 else "", color='b', marker='.')
            plt.scatter(target[0], target[1], label='Target' if i == 0 else "", color='c', marker='.')
    ax.set_title("Prection Trajectory")
    ax.legend(loc='upper right')
    fig.savefig(scenario+"/errorpng/errors_visualization_" + str(model_name) + ".pdf")
    
    #Plot smooth path
    fig=plt.figure()
    data=v.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
    plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
    plt.legend(['Target','Prediction'],loc='upper right')
    plt.xlabel("X-latitude")
    plt.ylabel("Y-longitude")
    plt.title(str(model_name)+" Prediction")
    fig.savefig(scenario+"/predictionpng/"+str(model_name)+"_locprediction.png")
    
    #Numerical Analysis
    Y_pre=pd.DataFrame(Y_pre)
    Y_test=pd.DataFrame(Y_test)
    df=pd.concat([Y_pre, Y_test], axis=1)
    df.columns=['Pre_x','Pre_y', 'Target_x','Target_y']
    df['Error']= ((df['Pre_x']-df['Target_x'])**2+(df['Pre_y']-df['Target_y'])**2)**(1/2)
    Analysis=pd.DataFrame(df['Error'].describe())
    
    
    
    print (Analysis)
    return Analysis

 
Analysis=Analysis(model,model_name,scenario,batch_size)
Analysis.to_csv(str(model_name)+'_describe.csv')



