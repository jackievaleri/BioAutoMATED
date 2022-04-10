import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.8550593558086541
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=18, n_estimators=100, subsample=1.0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
