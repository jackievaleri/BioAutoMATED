import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# Average CV score on the training set was: -0.15805904687059955
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=10, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.25, verbosity=0)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=2, max_features=0.6500000000000001, min_samples_leaf=9, min_samples_split=5, n_estimators=100, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
