# MM-Loc

This is the code repo for paper MM-Loc: Cross-sensor Indoor Smartphone Location Tracking using Multimodal Deep Neural Networks.

# Dataprocessing

Convert raw data to machine learning data, with a readme file for user.

# Train and test
To debug all models, the following commands can be used as example:

sensor baseline model:

python sensor_baseline.py --scenario="scenarioA" --hidden_size=128 --learning_rate=0.005 --batch_size=100 --epoch=100

overlap downsample sensor model:

python sensor_baseline.py --scenario="scenarioA" --hidden_size=128 --learning_rate=0.005 --batch_size=100 --epoch=100

wifi model:

python sensor_baseline.py --scenario="scenarioA" --wifi_input_size=102 --hidden_size=128 --learning_rate=0.005 --batch_size=100 --epoch=100

mmloc model:

python sensor_baseline.py --scenario="scenarioA" --wifi_input_size=102 --hidden_size=128 --learning_rate=0.005 --batch_size=100 --epoch=100
