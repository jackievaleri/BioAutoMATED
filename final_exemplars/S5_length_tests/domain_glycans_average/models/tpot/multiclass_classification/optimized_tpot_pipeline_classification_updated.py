import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.8965572435780681
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.9500000000000001, min_samples_leaf=11, min_samples_split=9, n_estimators=100, subsample=0.9000000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
