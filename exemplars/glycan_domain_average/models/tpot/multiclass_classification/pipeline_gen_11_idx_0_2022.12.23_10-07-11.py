import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9411393140079923
exported_pipeline = XGBClassifier(learning_rate=0.5, max_depth=6, min_child_weight=6, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
