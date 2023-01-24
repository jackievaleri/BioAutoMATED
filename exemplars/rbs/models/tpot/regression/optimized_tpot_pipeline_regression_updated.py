import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Average CV score on the training set was: -0.07042308893264233
exported_pipeline = XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=18, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.05, verbosity=0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
