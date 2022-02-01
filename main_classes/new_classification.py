def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.pipeline import make_pipeline, make_union
	from tpot.builtins import StackingEstimator
	from xgboost import XGBClassifier
	
	# Average CV score on the training set was: 0.9575757575757576
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=7, max_features=0.5, min_samples_leaf=4, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)),
	    XGBClassifier(learning_rate=0.1, max_depth=4, min_child_weight=18, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
	)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results