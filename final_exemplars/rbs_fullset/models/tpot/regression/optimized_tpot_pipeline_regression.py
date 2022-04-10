import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.03752364191138799
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.1, loss="epsilon_insensitive", tol=0.001)),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.8500000000000001, min_samples_leaf=6, min_samples_split=13, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=20, min_samples_split=6, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
