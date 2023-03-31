import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# Average CV score on the training set was: 0.8285991745569312
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=4, min_samples_split=18)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.6500000000000001, min_samples_leaf=15, min_samples_split=6, n_estimators=100, subsample=0.25)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
