#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:21:45 2022

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

d1=pd.read_csv("modelvari.csv")

d2=pd.read_csv("finetune.csv")

mean=np.array(d2.iloc[0,1:])
std=np.array(d2.iloc[1,1:])

name=d2.columns


group_labels = name

fig=plt.figure()


plt.plot(bin_edgeMTLocB[0:-1],cdfMTLocB,linestyle='-',label=str('train from scratch'),linewidth=0.7)
plt.plot(bin_edgeMTLocB_8plot[0:-1],cdfMTLocB_8plot,linestyle='-',label=str('100%'),linewidth=0.7)

plt.set_xticklabels(group_labels)


plt.xlim([0, 20])
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.grid(True)
plt.title(('Fine Tuning Test CDF'))
plt.legend(loc='lower right')
fig.savefig("Fine-tuning_CDF(test).pdf")
print('cdf plotted')
