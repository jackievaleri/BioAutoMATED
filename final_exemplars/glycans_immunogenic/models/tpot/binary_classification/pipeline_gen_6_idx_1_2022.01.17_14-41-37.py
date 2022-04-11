import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9492424242424242
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=1.0, max_depth=4, max_features=0.6500000000000001, min_samples_leaf=13, min_samples_split=15, n_estimators=100, subsample=0.9500000000000001)),
    StackingEstimator(estimator=MLPClassifier(alpha=0.0001, learning_rate_init=1.0)),
    XGBClassifier(learning_rate=1.0, max_depth=1, min_child_weight=5, n_estimators=100, n_jobs=1, subsample=0.8, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
