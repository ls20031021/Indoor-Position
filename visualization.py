#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tensorflow.compat.v1 import flags

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        


def choose_scenario(scenario):   
    del_all_flags(flags.FLAGS)
    if scenario=='A':
        #Define Parameters ScenarioA
        flags.DEFINE_string("scenario", default="scenarioA", help="select scenarioA or scenarioB")
        flags.DEFINE_integer("wifi_input_size", default="102", help="wifi rss feature numbers")
        flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
        flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
        flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
        flags.DEFINE_integer("epoch", default="100", help="training epochs")
        flags.DEFINE_string("model_name", default="mmloc_scenarioA_overlap", help="model name")
        FLAGS = flags.FLAGS
    elif scenario=='B':
    #Define Parameters ScenarioB
        flags.DEFINE_string("scenario", default="scenarioB", help="select scenarioA or scenarioB")
        flags.DEFINE_integer("wifi_input_size", default="750", help="wifi rss feature numbers")
        flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
        flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
        flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
        flags.DEFINE_integer("epoch", default="100", help="training epochs")
        flags.DEFINE_string("model_name", default="mmloc_scenarioB_overlap", help="model name")
        FLAGS = flags.FLAGS
    return FLAGS    

def get_imu_labels(df, pointsnumber, step):
    count=0
    label=0
    round=0
    a=np.zeros((1,2))
    #for num in range(0,2*(pointsnumber-2)+1):
    for i in df.iloc[:,0]:
        if  count==0:
            t=np.array([[i-1,label]])
            a=np.concatenate((a,t),axis=0)
            count=count+1
            label=label+1
            #num=num+1
            # if num==df.iloc[i,0]-step:
            #     label=label+1
            #     print('here1')
            
    
        elif 0 < count < pointsnumber-1:
            t=np.array([[i-step,label]])
            a=np.concatenate((a,t),axis=0)
            label=label+1
            t=np.array([[i+step,label]])
            a=np.concatenate((a,t),axis=0)
            count=count+1
            label=label+1
            if label==(2*(pointsnumber-2)+1):
                label=2*(pointsnumber-2)
    
        else:
            t=np.array([[i,label]])
            a=np.concatenate((a,t),axis=0)
            count=0
            label=0
    
            round=round+1
            print('Round '+str(round)+' finish')
    a=a[1::] #drop the first initial row of zeros
    
    cnt=0
    b=np.zeros((1,2))
    
    for i in range (0, int(a[-1,0])+1):
        if i == a[cnt,0]:
            tag=a[cnt,1]
            cnt=cnt+1
        tem=np.array([[i,tag]])
        b=np.concatenate((b,tem),axis=0)
    b=b[1::]    
    b=b[:,1]


    return b

def normalized_data_to_utm(dd):
#for scenarioa
    min_c1 = 0
    max_c1 = 39.76
    min_c2 = 0
    max_c2 = 32.63
#for scenariob
    # min_c1 = 0
    # max_c1 = 39.76
    # min_c2 = 0
    # max_c2 = 32.63
        
    d1 = dd[:, 0]
    d2 = dd[:, 1]
    d3 = dd[:, 2]
    d4 = dd[:, 3]
    
    inverse_to_utm_x = lambda x: (min_c1 + x * (max_c1 - min_c1))
    inverse_to_utm_y = lambda x: (min_c2 + x * (max_c2 - min_c2))
    
    id1 = inverse_to_utm_x(d1)
    id2 = inverse_to_utm_y(d2)
    id3 = inverse_to_utm_x(d3)
    id4 = inverse_to_utm_y(d4)
    
    return np.transpose(np.vstack((np.vstack((id1, id2)), np.vstack((id3, id4)))))

def cal_error_in_meters(data):
    data = normalized_data_to_utm(data)
    errors = [np.sqrt(np.square(item[0] - item[2]) + np.square(item[1] - item[3])) for item in data]
    return errors


def cdfpic(data):
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)
    
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    
    hist=hist.astype(float)/len(data)

    cdf = np.cumsum(hist)
    
    return bin_edges,cdf

def cdfdiff(target, predict):
    target_and_predict = np.hstack((target, predict))
    error_in_meters = cal_error_in_meters(target_and_predict)    
    return cdfpic(error_in_meters)

def get_ave_prediction(locPrediction, n):
    weights = np.ones(n)
    weights /= weights.sum()
    x = np.asarray(locPrediction[:,0])
    y = np.asarray(locPrediction[:,1])  
    avelatPrediction = np.convolve(x, weights, mode='full')[:len(x)]
    avelngPrediction = np.convolve(y, weights, mode='full')[:len(y)]
    avelatPrediction[:n] = avelatPrediction[n]
    avelngPrediction[:n] = avelngPrediction[n]
    avelatPrediction=avelatPrediction.reshape(-1,1)
    avelngPrediction=avelngPrediction.reshape(-1,1)
    aveLocPrediction=np.column_stack((avelatPrediction,avelngPrediction))
    return aveLocPrediction

def visualization(locationtest, locPrediction, suffix):
    data=normalized_data_to_utm(np.hstack((locationtest, locPrediction)))
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
        plt.plot([pred[0], target[0]], [pred[1], target[1]], color='r',
                 linewidth=0.5, label='Error Line' if i == 0 else "")
        plt.scatter(pred[0], pred[1], label='Prediction' if i == 0 else "", color='b', marker='.')
        plt.scatter(target[0], target[1], label='Target' if i == 0 else "", color='c', marker='.')
    ax.set_title("Prection Trajectory")
    ax.legend(loc='upper right')
    # save error line fig
    fig.savefig("errors_visualization_" + str(suffix) + ".pdf")
    print('error plotted')
    return fig

def draw_cdf_picture(locationtest,locPrediction,model_name,scenario):
    fig=plt.figure()
    bin_edge,cdf=cdfdiff(target=locationtest,predict=locPrediction)
    plt.plot(bin_edge[0:-1],cdf,linestyle='--',label=str(model_name),color='r')
    plt.xlim(xmin = 0)
    plt.ylim((0,1))
    plt.xlabel("metres")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.title((str(model_name)+' CDF'))
    plt.legend(loc='upper right')
    fig.savefig(scenario+"/cdf/"+str(model_name)+"_CDF.pdf")
    print('cdf plotted')
    return fig
    
def print_locprediction(locationtest,aveLocPrediction,model_name,scenario):
    fig=plt.figure()
    data=normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
    plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
    plt.legend(['Target','Prediction'],loc='upper right')
    plt.xlabel("X-latitude")
    plt.ylabel("Y-longitude")
    plt.title(str(model_name)+" Prediction")
    fig.savefig(scenario+"/predictionpng/"+str(model_name)+"_locprediction.png")
    print('trajectory plotted')
    return fig