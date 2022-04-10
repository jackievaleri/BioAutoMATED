import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9409090909090908
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    ),
    SelectFwe(score_func=f_classif, alpha=0.003),
    XGBClassifier(learning_rate=0.5, max_depth=4, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
