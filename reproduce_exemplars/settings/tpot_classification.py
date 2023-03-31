def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.svm import LinearSVC
	
	# Average CV score on the training set was: 1.0
	exported_pipeline = LinearSVC(C=20.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.0001)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results