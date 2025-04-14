# MM-Loc

This is the code repository for the paper:

**MM-Loc: Cross-sensor Indoor Smartphone Location Tracking using Multimodal Deep Neural Networks**

---

## 📁 Data Processing

Convert raw data into machine learning-ready format.

Refer to the included `README` file in the data processing directory for detailed usage instructions.

---

## 🧪 Train and Test

To debug or run all models, the following example commands can be used:

### 🔹 Sensor Baseline Model

```bash
python sensor_baseline.py --scenario="scenarioA" --hidden_size=128 --learning_rate=0.005 --batch_size=100 --epoch=100

