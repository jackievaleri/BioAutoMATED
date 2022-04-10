import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# Average CV score on the training set was: 0.8946666666666667
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=3, n_estimators=100, n_jobs=1, subsample=0.9000000000000001, verbosity=0)),
    VarianceThreshold(threshold=0.001),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.8, min_samples_leaf=6, min_samples_split=16, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
