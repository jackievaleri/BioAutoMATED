import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Average CV score on the training set was: -0.08015893926198388
exported_pipeline = XGBRegressor(learning_rate=1.0, max_depth=3, min_child_weight=13, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7500000000000001, verbosity=0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
