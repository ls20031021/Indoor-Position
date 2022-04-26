#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:29:54 2022

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


#Choose Scenario: Type A or B to select loading data.
FLAGS=v.choose_scenario('A')

wifi_input_size = FLAGS.wifi_input_size
hidden_size = FLAGS.hidden_size
batch_size = FLAGS.batch_size
epoch = FLAGS.epoch
learning_rate = FLAGS.learning_rate
scenario=FLAGS.scenario
model_name = FLAGS.model_name

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

#Load trained model
model=load_model(str(scenario)+"/model/"+str(model_name)+".h5")

# for i in range (0,len(model.layers)):
#     print (i, model.layers[i])

#extract lstm layer from trained model    
lstm_extracted=model.get_layer('lstm')
lstm_extracted.trainable = False #make transfered layer non-trainable

#build new model with transfer learning
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensoroutput=lstm_extracted(sensorinput) #use the trained lstm layer without updating the parameters

wifiinput=Input(shape=(wifi_input_size,))
wifi=Dense(hidden_size)(wifiinput)
wifi=ReLU()(wifi)
wifi=Dense(hidden_size)(wifi)
wifi=ReLU()(wifi)
wifioutput=Dense(hidden_size)(wifi)

merge=concatenate([sensoroutput,wifioutput])
hidden=Dense(hidden_size,activation='relu')(merge)
output=Dense(2,activation='relu')(hidden)
mmloc=Model(inputs=[sensorinput,wifiinput],outputs=[output])

mmloc.compile(optimizer=RMSprop(learning_rate),
             loss='mse',metrics=['acc'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

mmloc.fit([SensorTrain,WifiTrain], locationtrain,
                   validation_data=([SensorVal,WifiVal],locationval),
                   epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                   #shuffle=False,
                   )

#save model
mmloc.save(scenario+"/model/"+str(model_name)+"_transfer.h5")

locPrediction = mmloc.predict([SensorTest,WifiTest], batch_size=batch_size)
aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
#visualization for error line and location prediction
v.visualization(locationtest,locPrediction,model_name)
#print location prediction picture
v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
#draw cdf picture
v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
if __name__ == "__main__":
    tf.app.run()