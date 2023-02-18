import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.956060606060606
exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=6, max_features=0.9000000000000001, min_samples_leaf=10, min_samples_split=16, n_estimators=100, subsample=0.9500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
