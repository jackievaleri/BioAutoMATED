import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

# Average CV score on the training set was: 0.90625
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.0001),
    MaxAbsScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.5, min_samples_leaf=14, min_samples_split=14, n_estimators=100, subsample=0.6000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
