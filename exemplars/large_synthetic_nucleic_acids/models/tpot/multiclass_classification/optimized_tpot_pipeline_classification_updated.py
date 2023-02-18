import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Average CV score on the training set was: 0.8775999999999999
exported_pipeline = LinearSVC(C=1.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.01)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
