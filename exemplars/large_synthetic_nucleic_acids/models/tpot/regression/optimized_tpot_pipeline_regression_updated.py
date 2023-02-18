import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: -1.624336172947541e-29
exported_pipeline = LassoLarsCV(normalize=False)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
