import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# Average CV score on the training set was: 0.9530303030303029
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            make_pipeline(
                StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=True)),
                VarianceThreshold(threshold=0.0001)
            )
        ),
        FunctionTransformer(copy)
    ),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.6500000000000001, min_samples_leaf=1, min_samples_split=18, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
