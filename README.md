**Deep Learning for Macroeconomic Forecasting**

This project implements and compares **custom-built deep learning architectures** (vanilla Transformer and LSTM) against **library-based models** for multivariate, multi-horizon macroeconomic forecasting, with a focus on predicting inflation dynamics using U.S. economic indicators.

---

**Overview**

The pipeline builds a complete forecasting system:

* Retrieves macroeconomic time-series data from the FRED API
* Applies economic transformations (log-differences, rate changes)
* Constructs sliding window sequences for supervised learning
* Trains and compares:
  * Custom LSTM (from scratch)
  * Custom vanilla Transformer (from scratch)
  * Library-based LSTM/RNN models
  * Statistical baseline (VAR)
* Evaluates performance across multiple forecasting horizons

The primary objective is to analyze whether **custom implementations of sequence models** can match or outperform **library-optimized architectures**, and to understand trade-offs in learning dynamics, generalization, and stability in macroeconomic time-series settings.

---

**Project Structure**

```bash
deep_learning_project/
│
├── DL_project_macroeconomic_forecasting_final.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── trainer.py
│   ├── evaluation.py
│   ├── visuals.py
│
│   ├── model_var.py
│   ├── model_library_rnn.py
│   ├── model_custom_lstm.py
│   ├── model_custom_transformer.py
│   ├── model_tft.py
```

---

**Features**

**Data Pipeline**

* Data is retrieved using `fredapi`
* Macroeconomic indicators include:
  * CPI (Inflation)
  * Federal Funds Rate
  * GDP
  * Unemployment
  * Industrial Production

**Feature Engineering**

* Converts raw time-series into stationary representations:
  * Log differences for growth-based variables
  * First differences for rate-based variables

**Sequence Construction**

* Sliding window input sequences (default: 24 timesteps)
* Multi-horizon prediction targets (default: 6 steps ahead)

**Models Implemented**

* Custom LSTM (manual implementation using PyTorch)
* Custom Transformer (vanilla encoder architecture)
* Library LSTM/RNN (PyTorch built-in modules)
* VAR (statistical baseline)

**Training Framework**

* PyTorch training pipeline
* Mean Squared Error (MSE) loss
* Gradient clipping
* Early stopping

---

**Installation**

```bash
pip install pandas numpy torch scikit-learn fredapi matplotlib
```

---

**FRED API Setup**

You need a FRED API key:

* Register at: https://fred.stlouisfed.org/

Use it in your code:

```python
from src.data_utils import get_processed_data

df = get_processed_data(api_key="YOUR_API_KEY")
```

---

**How to Run**

**Option 1: Notebook (Recommended)**

```bash
DL_project_macroeconomic_forecasting_final.ipynb
```

This will execute:

* Data loading
* Preprocessing
* Model training
* Evaluation
* Visualization

---

**Option 2: Script-Based Workflow**

```python
from src.data_utils import get_processed_data, CustomDataProcessor
from src.trainer import train_custom_model
```

Typical workflow:

* Load data
* Prepare dataloaders
* Initialize model
* Train
* Evaluate

---

**Training Details**

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Gradient clipping (`max_norm=1.0`)
* Early stopping

---

**Evaluation**

Models are evaluated using:

* Validation loss (MSE)
* Multi-horizon forecasting accuracy
* Stability across prediction windows

---

**Key Insights**

* Library LSTM models are strong baselines
* Custom LSTM can be competitive with tuning
* Custom Transformers are sensitive to data size and hyperparameters
* VAR provides a useful classical benchmark

---

**Limitations**

* Small dataset size
* Limited hyperparameter tuning
* Transformer underperformance in low-data regimes
* Stationarity assumptions

---

**Future Work**

* Better hyperparameter optimization
* More macroeconomic indicators
* Probabilistic forecasting
* Hybrid econometric + deep learning models

---

**Contributing**

* Add new models
* Improve training pipeline
* Extend evaluation metrics

---

**License**

Academic and research use only.
