import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# Average CV score on the training set was: 0.921283017086066
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=4, min_samples_split=20)),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=7, max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=19, n_estimators=100, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
