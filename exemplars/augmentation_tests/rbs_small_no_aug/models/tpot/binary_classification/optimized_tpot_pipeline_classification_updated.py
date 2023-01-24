import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder

# Average CV score on the training set was: 0.9259999999999999
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.25, min_samples_leaf=6, min_samples_split=17, n_estimators=100, subsample=0.8500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
