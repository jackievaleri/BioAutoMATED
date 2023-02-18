import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Average CV score on the training set was: 0.8339960670901025
exported_pipeline = MLPClassifier(alpha=0.1, learning_rate_init=0.001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
