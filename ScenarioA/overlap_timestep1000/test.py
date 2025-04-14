import numpy as np
import os


file_path = os.path.join("ScenarioA", "overlap_timestep1000", "sensor_baseline_val.npy")
data = np.load(file_path)

print(data.shape)
