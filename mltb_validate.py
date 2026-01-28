import mltable

train_tbl = mltable.load("./data/diamond_train")
test_tbl = mltable.load("./data/diamond_test")

print(train_tbl.to_pandas_dataframe().shape)
print(test_tbl.to_pandas_dataframe().shape)
