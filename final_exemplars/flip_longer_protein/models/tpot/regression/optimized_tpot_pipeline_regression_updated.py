import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: -0.1501460970754273
exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=1.0, loss="huber", max_depth=1, max_features=0.2, min_samples_leaf=16, min_samples_split=4, n_estimators=100, subsample=0.6000000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
