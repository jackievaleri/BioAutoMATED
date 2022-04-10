import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.33006229561423966
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.8, min_samples_leaf=14, min_samples_split=14, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
