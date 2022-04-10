import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8679161354299476
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=0.1, dual=False, loss="squared_hinge", penalty="l2", tol=0.001)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=3, min_samples_split=15)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.9000000000000001, min_samples_leaf=2, min_samples_split=16, n_estimators=100, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
