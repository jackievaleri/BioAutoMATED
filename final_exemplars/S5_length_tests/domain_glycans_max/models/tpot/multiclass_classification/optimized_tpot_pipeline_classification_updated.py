import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Average CV score on the training set was: 0.8990825522402233
exported_pipeline = XGBClassifier(learning_rate=0.5, max_depth=6, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
