# Diamond Carat Prediction:"diamond_rf_pipeline"
Azure ML SDK v2 pipeline with registered components to predict diamond carat using RandomForestRegressor.

# Setup
bash setup.sh


#Update subscription_id in Python file pipeline.py.

# Submit pipeline
python pipeline.py
Using Your Resources

Resource Group: rg-dp100
Workspace: mlws-dp100
Data: diamonds_data (version 1)
Environment: diamonds_env (version:1)
Compute: aml-compute

# MLTable
Conversion and handeling of csv to parquet format.
validate loading and conversion to pandas dataframe locally.

# pipeline 
diamonds_data(csv to mltable) → Preprocess → Train → Evaluate → Model + Metrics
preprocess_diamonds - Encodes categoricals, prepare data for training step
train_randomForestRegressor_diamonds - Trains with MLflow  infer_signature
evaluate_model - Calculates RMSE, MAE, R²

