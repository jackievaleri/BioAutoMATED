import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Average CV score on the training set was: 0.3854417452113439
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.05),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=3, min_samples_split=9, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
