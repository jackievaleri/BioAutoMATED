import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9125
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.45, min_samples_leaf=3, min_samples_split=18, n_estimators=100)),
        FunctionTransformer(copy)
    ),
    XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=2, n_estimators=100, n_jobs=1, subsample=0.8, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
