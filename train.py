import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--model_output", type=str)
    args = parser.parse_args()

    # Load preprocessed train/test parquet files
    train_df = pd.read_parquet(os.path.join(args.train_data, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(args.test_data, "test.parquet"))

    target_col = "carat"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Define model (no need for pipeline since preprocessing is already done)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        signature = mlflow.models.signature.infer_signature(X_test, preds)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )

    # Save trained model
    os.makedirs(args.model_output, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_output, "model.joblib"))


if __name__ == "__main__":
    main()
