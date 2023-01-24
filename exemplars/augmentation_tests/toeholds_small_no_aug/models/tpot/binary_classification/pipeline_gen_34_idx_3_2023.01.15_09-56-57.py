import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8304147869674186
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=6, min_child_weight=8, n_estimators=100, n_jobs=1, subsample=0.6000000000000001, verbosity=0)),
    StackingEstimator(estimator=BernoulliNB(alpha=10.0, fit_prior=True)),
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=24, p=2, weights="uniform")),
    MaxAbsScaler(),
    LinearSVC(C=0.1, dual=False, loss="squared_hinge", penalty="l2", tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
