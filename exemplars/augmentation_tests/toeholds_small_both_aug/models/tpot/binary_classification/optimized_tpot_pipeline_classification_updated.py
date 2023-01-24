import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.8180754795663052
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.4, min_samples_leaf=6, min_samples_split=18, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
