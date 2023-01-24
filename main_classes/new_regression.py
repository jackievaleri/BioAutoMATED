def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.model_selection import train_test_split
	
	# Average CV score on the training set was: -0.3658711306850905
	exported_pipeline = RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=5, min_samples_split=2, n_estimators=100)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results