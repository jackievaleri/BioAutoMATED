import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# Average CV score on the training set was: 0.5679200972146929
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=7, min_samples_split=17, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
