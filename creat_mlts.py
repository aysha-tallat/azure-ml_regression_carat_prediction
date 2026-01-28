import mltable

train_tbl = mltable.load("diamonds_train")
test_tbl = mltable.load("diamonds_test")

train_df = train_tbl.to_pandas_dataframe()
test_df = test_tbl.to_pandas_dataframe()

print(len(train_df), len(test_df))
print(train_df.columns)
