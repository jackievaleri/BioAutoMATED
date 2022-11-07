import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoLarsCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -7.504323435581672e-32
exported_pipeline = make_pipeline(
    FastICA(tol=0.30000000000000004),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=True, l1_ratio=0.5, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=0.1)),
    RobustScaler(),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
