import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow.compat.v1 as tf
import os  
import visualization as v
from absl import flags

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, concatenate, LSTM, Input, ReLU, Layer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, TensorBoard
from tensorflow.keras import backend as K

np.random.seed(7)


class DynamicWeightedFusion(Layer):
    
    def __init__(self, **kwargs):
        super(DynamicWeightedFusion, self).__init__(**kwargs)
        
    def build(self, input_shape):
       
        self.wifi_weight = self.add_weight(name='wifi_weight', shape=(1,), initializer='ones', trainable=True)
        self.sensor_weight = self.add_weight(name='sensor_weight', shape=(1,), initializer='ones', trainable=True)
        super(DynamicWeightedFusion, self).build(input_shape)
        
    def call(self, inputs):
        sensor_feat, wifi_feat = inputs
        
       
        wifi_quality = 1.0 / (K.var(wifi_feat, axis=1, keepdims=True) + 1e-8)
        sensor_quality = 1.0 / (K.var(sensor_feat, axis=1, keepdims=True) + 1e-8)
        
       
        total_weight = (self.wifi_weight * wifi_quality + 
                      self.sensor_weight * sensor_quality)
        wifi_ratio = (self.wifi_weight * wifi_quality) / total_weight
        sensor_ratio = (self.sensor_weight * sensor_quality) / total_weight
        
       
        weighted_wifi = wifi_feat * wifi_ratio
        weighted_sensor = sensor_feat * sensor_ratio
        return K.concatenate([weighted_sensor, weighted_wifi], axis=-1)


# Hyper-parameters
flags.DEFINE_string("scenario", default="scenarioA", help="select scenarioA or scenarioB")
flags.DEFINE_integer("wifi_input_size", default="102", help="wifi rss feature numbers")
flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
flags.DEFINE_integer("epoch", default="100", help="training epochs")
FLAGS = flags.FLAGS

def preprocess_wifi_data(wifi_data, wifi_input_size):
    sorted_wifi = np.sort(wifi_data, axis=1)[:, :wifi_input_size]
    normalized_wifi = (sorted_wifi - np.min(sorted_wifi, axis=1, keepdims=True)) / \
                      (np.ptp(sorted_wifi, axis=1, keepdims=True) + 1e-8)
    return normalized_wifi

def save_error_statistics(locationtest, predictions, wifi_input_size, filepath):    
    errors = np.sqrt(np.sum((locationtest - predictions) ** 2, axis=1))    
    statistics = {
        'wifi_input_size': wifi_input_size,
        'min': np.min(errors),
        'max': np.max(errors),
        'mean': np.mean(errors),
        'std_dev': np.std(errors)
    }    
    df = pd.DataFrame([statistics])    
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)

def main(_):
    scenario = FLAGS.scenario
    wifi_input_size = FLAGS.wifi_input_size
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    learning_rate = FLAGS.learning_rate

    model_name = "mmloc_scenarioA_overlap"

    # Load and preprocess data
    SensorTrain = np.load(scenario + "/overlap_timestep1000/overlap_ds_sensor_train.npy")
    locationtrain = np.load(scenario + "/overlap_timestep1000/overlap_ds_location_train.npy")
    WifiTrain = preprocess_wifi_data(np.load(scenario + "/overlap_timestep1000/overlap_ds_wifi_train.npy"), wifi_input_size)

    SensorVal = np.load(scenario + "/overlap_timestep1000/overlap_ds_sensor_val.npy")
    locationval = np.load(scenario + "/overlap_timestep1000/overlap_ds_location_val.npy")
    WifiVal = preprocess_wifi_data(np.load(scenario + "/overlap_timestep1000/overlap_ds_wifi_val.npy"), wifi_input_size)

    SensorTest = np.load(scenario + "/overlap_timestep1000/overlap_ds_sensor_test.npy")
    locationtest = np.load(scenario + "/overlap_timestep1000/overlap_ds_location_test.npy")
    WifiTest = preprocess_wifi_data(np.load(scenario + "/overlap_timestep1000/overlap_ds_wifi_test.npy"), wifi_input_size)

    # Construct mmloc model (仅修改融合层)
    sensorinput = Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
    sensoroutput = LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]), units=hidden_size)(sensorinput)

    wifiinput = Input(shape=(wifi_input_size,))
    wifi = Dense(hidden_size)(wifiinput)
    wifi = ReLU()(wifi)
    wifi = Dense(hidden_size)(wifi)
    wifi = ReLU()(wifi)
    wifioutput = Dense(hidden_size)(wifi)

    # ============== 唯一修改行：替换concatenate为动态融合 ==============
    merge = DynamicWeightedFusion()([sensoroutput, wifioutput])  # Changed line
    
    hidden = Dense(hidden_size, activation='relu')(merge)
    output = Dense(2, activation='relu')(hidden)
    mmloc = Model(inputs=[sensorinput, wifiinput], outputs=[output])

    mmloc.compile(optimizer=RMSprop(learning_rate),
                  loss='mse', metrics=['acc'])

    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    # Train model
    mmloc.fit([SensorTrain, WifiTrain], locationtrain,
              validation_data=([SensorVal, WifiVal], locationval),
              epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[tensorboard])

    # Save model
    mmloc.save(scenario + "/model/" + str(model_name) + ".h5")

    # Prediction
    locPrediction = mmloc.predict([SensorTest, WifiTest], batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)

    # 保存误差统计到 CSV 文件
    save_error_statistics(locationtest, locPrediction, wifi_input_size, 
                          "C:/Users/Administrator/Desktop/MMLoc/error_statistics.csv")

    # Debug: Check data shapes and ranges
    print("Location Test Shape:", locationtest.shape)
    print("Location Prediction Shape:", locPrediction.shape)
    print("Averaged Location Prediction Shape:", aveLocPrediction.shape)
    print("Location Prediction Min:", np.min(locPrediction))
    print("Location Prediction Max:", np.max(locPrediction))

    # Visualization
    try:
        v.visualization(locationtest, locPrediction, model_name)
        v.print_locprediction(locationtest, aveLocPrediction, model_name, scenario)
        v.draw_cdf_picture(locationtest, locPrediction, model_name, scenario)
    except Exception as e:
        print("Error in visualization:", str(e))

if __name__ == "__main__":
    tf.app.run()