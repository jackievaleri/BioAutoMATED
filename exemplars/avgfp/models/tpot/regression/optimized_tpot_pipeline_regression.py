import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.04894333268292335
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=7, min_samples_split=5, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="ls", max_depth=2, max_features=1.0, min_samples_leaf=10, min_samples_split=13, n_estimators=100, subsample=0.25)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
