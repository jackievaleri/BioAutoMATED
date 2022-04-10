import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# Average CV score on the training set was: 0.8748585626018691
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=True, l1_ratio=1.0, learning_rate="invscaling", loss="modified_huber", penalty="elasticnet", power_t=50.0)),
    MinMaxScaler(),
    XGBClassifier(learning_rate=0.5, max_depth=5, min_child_weight=3, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
