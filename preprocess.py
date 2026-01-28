import argparse
import os
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
import mltable



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--train_output", type=str)
    parser.add_argument("--test_output", type=str)
    args = parser.parse_args()

    train_df = mltable.load(args.train_data).to_pandas_dataframe()
    test_df = mltable.load(args.test_data).to_pandas_dataframe()
    target_col = "carat"


    categorical_features = ["cut", "color", "clarity"]
    numerical_features = [
    "depth_percent",
    "table",
    "price",
    "length",
    "width",
    "depth",
    ]


    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order = ["J", "I", "H", "G", "F", "E", "D"]
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


    ordinal_encoder = OrdinalEncoder(
        categories=[cut_order, color_order, clarity_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        )


    preprocessor = ColumnTransformer(
        transformers=[
        ("cat", ordinal_encoder, categorical_features),
        ("num", "passthrough", numerical_features),
        ]
        )


    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]


    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]


    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)


    
    train_processed = pd.DataFrame(X_train_transformed)
    train_processed[target_col] = y_train.values


    test_processed = pd.DataFrame(X_test_transformed)
    test_processed[target_col] = y_test.values


    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)


    train_processed.to_parquet(os.path.join(args.train_output, "train.parquet"))
    test_processed.to_parquet(os.path.join(args.test_output, "test.parquet"))




if __name__ == "__main__":
    main()