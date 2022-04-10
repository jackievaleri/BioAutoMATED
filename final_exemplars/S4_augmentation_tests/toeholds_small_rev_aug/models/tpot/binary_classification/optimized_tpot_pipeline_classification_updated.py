import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# Average CV score on the training set was: 0.8166548811013767
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)),
    VarianceThreshold(threshold=0.001),
    DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=19, min_samples_split=19)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
