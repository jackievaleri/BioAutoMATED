import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# Average CV score on the training set was: -8.138672999895571e-32
exported_pipeline = make_pipeline(
    FastICA(tol=1.0),
    RobustScaler(),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
