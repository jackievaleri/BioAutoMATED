import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Average CV score on the training set was: -8.101281070025294e-32
exported_pipeline = make_pipeline(
    FastICA(tol=0.55),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
