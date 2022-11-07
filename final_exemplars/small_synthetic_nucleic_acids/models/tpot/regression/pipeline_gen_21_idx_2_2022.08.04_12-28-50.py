import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoLarsCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -7.845027664981066e-32
exported_pipeline = make_pipeline(
    FastICA(tol=0.45),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=0.5)),
    VarianceThreshold(threshold=0.0005),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
