import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from tpot.builtins import OneHotEncoder, StackingEstimator

# Average CV score on the training set was: 0.8309147869674186
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=False)),
    Normalizer(norm="l1"),
    LinearSVC(C=20.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.0001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
