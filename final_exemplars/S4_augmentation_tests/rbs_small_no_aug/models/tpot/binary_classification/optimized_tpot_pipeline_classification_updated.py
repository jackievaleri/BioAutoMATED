import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# Average CV score on the training set was: 0.9164999999999999
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=0.0, learning_rate="invscaling", loss="hinge", penalty="elasticnet", power_t=100.0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.45, min_samples_leaf=3, min_samples_split=13, n_estimators=100, subsample=0.7500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
