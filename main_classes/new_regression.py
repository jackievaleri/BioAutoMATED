def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
	from sklearn.model_selection import train_test_split
	from sklearn.pipeline import make_pipeline, make_union
	from tpot.builtins import StackingEstimator
	
	# Average CV score on the training set was: -0.31957709463134315
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=17, min_samples_split=11, n_estimators=100)),
	    RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=2, min_samples_split=19, n_estimators=100)
	)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results