import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# Average CV score on the training set was: 0.8206586357947433
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=20, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
