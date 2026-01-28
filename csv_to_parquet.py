import pandas as pd
from sklearn.model_selection import train_test_split
import os

# paths
os.makedirs("data", exist_ok=True)

# load csv
df = pd.read_csv("./diamond/diamonds_data/diamonds.csv")

# deterministic split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# write parquet
train_df.to_parquet("data/diamonds_train.parquet", index=False)
test_df.to_parquet("data/diamonds_test.parquet", index=False)

print("Parquet files created:")
print(" - data/diamonds_train.parquet")
print(" - data/diamonds_test.parquet")
