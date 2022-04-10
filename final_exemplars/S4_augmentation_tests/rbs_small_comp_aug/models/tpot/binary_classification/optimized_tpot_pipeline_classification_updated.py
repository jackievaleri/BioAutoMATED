import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator

# Average CV score on the training set was: 0.9065000000000001
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=True, l1_ratio=0.0, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=0.5)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.9000000000000001, min_samples_leaf=14, min_samples_split=5, n_estimators=100, subsample=0.55)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
