#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:41:39 2022

@author: weixijia
"""
import numpy as np
import matplotlib.pyplot as plt

#A=[1.6, 2.2, 2.7, 3.0, 3.4, 5.0] #mean
#B=[1.6, 2.2, 2.8, 3.2, 3.6, 5.2] #mean
#C=[1.6, 2.6, 3.3, 4.1, 4.6, 7.2] #mean

A=[1.05,	1.25,	1.39,	1.56,	1.67,	2.33] #median error
B=[1.05,	1.13,	1.22	,1.30	,1.38	,1.71] #median error
C=[1.05	,1.26	,1.48,	1.81,	2.05	,4.60] #median error

x=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


plt.plot(x,A, label='IMU')
for a,b in zip(x, A): 
    plt.text(a, b, str(b), fontsize=5, verticalalignment='top', horizontalalignment='center')
plt.plot(x,B, label='Mag')
for a,b in zip(x, B): 
    plt.text(a, b, str(b), fontsize=5, verticalalignment='top', horizontalalignment='center')
plt.plot(x,C, label='WiFi')
for a,b in zip(x, C): 
    plt.text(a, b, str(b), fontsize=5, verticalalignment='bottom', horizontalalignment='center')

plt.xlabel("Percentage")
plt.ylabel("Metres")
plt.legend(loc='lower right')
plt.title("Prediction Median Error")
plt.grid()
plt.savefig('sheltering_median.pdf')
plt.show()
