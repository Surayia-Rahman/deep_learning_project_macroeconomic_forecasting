📊 Deep Learning for Macroeconomic Forecasting

This project implements and compares multiple deep learning and statistical models for multivariate, multi-horizon macroeconomic forecasting, with a focus on predicting inflation dynamics using U.S. economic indicators.

Overview

The pipeline builds a complete forecasting system:

- Retrieves macroeconomic time-series data from the FRED API
- Applies economic transformations (log-differences, rate changes)
- Constructs sliding window sequences for supervised learning
- Trains multiple models (RNNs, LSTMs, Transformers, TFT, VAR)
- Evaluates performance across forecasting horizons

The primary objective is to assess whether modern deep learning architectures outperform traditional approaches in structured macroeconomic forecasting tasks.

📁 Project Structure
deep_learning_project/
│
├── DL_project_macroeconomic_forecasting_final.ipynb   # Main experiment notebook
│
├── src/
│   ├── data_utils.py                # Data retrieval, preprocessing, loaders
│   ├── trainer.py                  # Training loop with early stopping
│   ├── evaluation.py               # Metrics and evaluation logic
│   ├── visuals.py                  # Plotting and visualization utilities
│
│   ├── model_var.py                # Vector Autoregression (baseline)
│   ├── model_library_rnn.py        # Library-based RNN/LSTM
│   ├── model_custom_lstm.py        # Custom LSTM implementation
│   ├── model_custom_transformer.py # Custom Transformer model
│   ├── model_tft.py                # Temporal Fusion Transformer (TFT)
📦 Features
1. Data Pipeline
Data sourced using fredapi
Indicators include:
CPI (Inflation)
Federal Funds Rate
GDP
Unemployment
Industrial Production
2. Feature Engineering

Transforms raw macro data into stationary signals:

Log differences (e.g., inflation, GDP growth)
First differences (e.g., interest rates, unemployment)
3. Sequence Construction
Sliding window input (default: 24 timesteps)
Multi-step forecasting horizon (default: 6 steps ahead)
4. Models Implemented
VAR (baseline statistical model)
Library LSTM/RNN
Custom LSTM
Custom Transformer
Temporal Fusion Transformer (TFT)
5. Training Framework
PyTorch-based training loop
MSE loss
Gradient clipping
Early stopping with patience
⚙️ Installation
pip install pandas numpy torch scikit-learn fredapi matplotlib
🔑 FRED API Setup

You need a FRED API key:

Register at: https://fred.stlouisfed.org/
Use it in your code:
from src.data_utils import get_processed_data

df = get_processed_data(api_key="YOUR_API_KEY")
▶️ How to Run
Option 1: Notebook (Recommended)

Run:

DL_project_macroeconomic_forecasting_final.ipynb

This executes:

Data loading
Model training
Evaluation
Visualization
Option 2: Script-Based Workflow
from src.data_utils import get_processed_data, CustomDataProcessor
from src.trainer import train_custom_model

Typical flow:

Load data
Prepare dataloaders
Initialize model
Train
Evaluate
🧠 Training Details
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
Regularization:
Gradient clipping (max_norm=1.0)
Early stopping
📏 Evaluation

Models are evaluated on:

Validation loss (MSE)
Multi-horizon prediction accuracy
Forecast stability
📊 Key Insights
LSTM-based models perform strongly on structured macroeconomic data
Transformer-based models require careful tuning and may not outperform RNNs in low-sample regimes
Classical methods (VAR) provide a strong baseline but lack flexibility for nonlinear dynamics
⚠️ Limitations
Limited dataset size (typical in macroeconomics)
No exogenous covariates beyond core indicators
Transformer/TFT models may be under-optimized
Assumes stationarity after transformations
🔮 Future Work
Incorporate additional macro/financial indicators
Hyperparameter optimization (especially for Transformers)
Probabilistic forecasting (uncertainty quantification)
Hybrid physics-informed or econometric models
Real-time forecasting pipeline
🤝 Contributing

Contributions are welcome. You can:

Add new models
Improve training stability
Extend evaluation metrics
Optimize data preprocessing
📜 License

This project is intended for academic and research use.
