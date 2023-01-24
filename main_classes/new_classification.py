def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import MaxAbsScaler
	
	# Average CV score on the training set was: 0.9181666666666667
	exported_pipeline = make_pipeline(
	    MaxAbsScaler(),
	    MaxAbsScaler(),
	    GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.35000000000000003, min_samples_leaf=20, min_samples_split=13, n_estimators=100, subsample=0.8500000000000001)
	)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results