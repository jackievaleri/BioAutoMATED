import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# Average CV score on the training set was: 1.0
exported_pipeline = make_pipeline(
    PCA(iterated_power=1, svd_solver="randomized"),
    LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
