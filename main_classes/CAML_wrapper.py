# Point to location of automl classes
import shutil
from datetime import datetime
from subprocess import call

#Ignore future warnings
import warnings
warnings.filterwarnings("ignore")

# Core Functions
from CAML_generic_automl_classes import * 
from CAML_generic_autokeras import AutoKerasClassification, AutoKerasRegression 
from CAML_generic_deepswarm import DeepSwarmClassification, DeepSwarmRegression
from CAML_generic_tpot import TPOTClassification, TPOTRegression 

# Enable multiprocessing in Forkserver mode
import multiprocessing
multiprocessing.set_start_method('forkserver')

# Helper function to copy directory for backups
def copy_full_dir(source, target):
    call(['cp', '-a', source, target]) # Unix


def run_bioseqml_binaryclass(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, do_auto_bin, bin_threshold, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params):
	data_path = data_folder + data_file 

	print("#################################################################################################")
	print("##############################            RUNNING DEEPSWARM           ###########################")
	print("#################################################################################################")

	# DeepSwarm (run)
	#   AutoML method Based on: 
	#      Byla, E. and Pang, W., 2019. DeepSwarm: Optimising Convolutional 
	#      Neural Networks using Swarm Intelligence. arXiv preprint arXiv:1905.07350.
	#   Script adapted from:
	#      https://github.com/Pattio/DeepSwarm

	# Define run folder path to store generated models and data
	run_folder = 'deepswarm/binary_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	dsc = DeepSwarmClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, yaml_params=yaml_params, num_final_epochs=num_final_epochs, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=False, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = dsc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING AUTOKERAS           ###########################")
	print("#################################################################################################")
	# AutoKeras (run)
	#   AutoML method Based on: 
	#      Jin, H., Song, Q. and Hu, X., 2019, July. Auto-keras: An efficient neural architecture search system. 
	#      In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & 
	#      Data Mining (pp. 1946-1956). ACM.
	#   Script adapted from:
	#      https://github.com/keras-team/autokeras

	## Define run folder path to store generated models and data
	run_folder = 'autokeras/binary_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	# Autokeras execution
	akc = AutoKerasClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=False, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = akc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING TPOT                ###########################")
	print("#################################################################################################")

	# TPOT (run)
	#   AutoML method Based on: 
	#      Olson, R.S., Bartley, N., Urbanowicz, R.J. and Moore, J.H., 2016, July. Evaluation of a tree-based 
	#      pipeline optimization tool for automating data science. In Proceedings of the Genetic and Evolutionary 
	#      Computation Conference 2016 (pp. 485-492). ACM.
	#   Script adapted from:
	#      https://github.com/EpistasisLab/tpot

	## Define run folder path to store generated models and data
	run_folder = 'tpot/binary_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	tpc = TPOTClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, population_size=population_size, num_generations=num_generations, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=False, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = tpc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

def run_bioseqml_multiclass(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, do_auto_bin, bin_threshold, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params):
	data_path = data_folder + data_file 

	print("#################################################################################################")
	print("##############################            RUNNING DEEPSWARM           ###########################")
	print("#################################################################################################")

	# DeepSwarm (run)
	#   AutoML method Based on: 
	#      Byla, E. and Pang, W., 2019. DeepSwarm: Optimising Convolutional 
	#      Neural Networks using Swarm Intelligence. arXiv preprint arXiv:1905.07350.
	#   Script adapted from:
	#      https://github.com/Pattio/DeepSwarm

	## Define run folder path to store generated models and data
	run_folder = 'deepswarm/multiclass_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	dsc = DeepSwarmClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, yaml_params=yaml_params, num_final_epochs=num_final_epochs, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=True, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = dsc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING AUTOKERAS           ###########################")
	print("#################################################################################################")
	# AutoKeras (run)
	#   AutoML method Based on: 
	#      Jin, H., Song, Q. and Hu, X., 2019, July. Auto-keras: An efficient neural architecture search system. 
	#      In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & 
	#      Data Mining (pp. 1946-1956). ACM.
	#   Script adapted from:
	#      https://github.com/keras-team/autokeras

	## Define run folder path to store generated models and data
	run_folder = 'autokeras/multiclass_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	# Autokeras execution
	akc = AutoKerasClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=True, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = akc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING TPOT                ###########################")
	print("#################################################################################################")

	# TPOT (run)
	#   AutoML method Based on: 
	#      Olson, R.S., Bartley, N., Urbanowicz, R.J. and Moore, J.H., 2016, July. Evaluation of a tree-based 
	#      pipeline optimization tool for automating data science. In Proceedings of the Genetic and Evolutionary 
	#      Computation Conference 2016 (pp. 485-492). ACM.
	#   Script adapted from:
	#      https://github.com/EpistasisLab/tpot

	## Define run folder path to store generated models and data
	run_folder = 'tpot/multiclass_classification/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	tpc = TPOTClassification(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, do_auto_bin=do_auto_bin, bin_threshold=bin_threshold, verbosity=verbosity, population_size=population_size, num_generations=num_generations, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, multiclass=True, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = tpc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

def run_bioseqml_regression(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params):
	data_path = data_folder + data_file 
	
	print("#################################################################################################")
	print("##############################            RUNNING DEEPSWARM           ###########################")
	print("#################################################################################################")

	# DeepSwarm (run)
	#   AutoML method Based on: 
	#      Byla, E. and Pang, W., 2019. DeepSwarm: Optimising Convolutional 
	#      Neural Networks using Swarm Intelligence. arXiv preprint arXiv:1905.07350.
	#   Script adapted from:
	#      https://github.com/Pattio/DeepSwarm

	## Define run folder path to store generated models and data
	run_folder = 'deepswarm/regression/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	dsc = DeepSwarmRegression(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, verbosity=verbosity, yaml_params=yaml_params, num_final_epochs=num_final_epochs, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
	final_model, transform_obj, results = dsc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING AUTOKERAS           ###########################")
	print("#################################################################################################")
	# AutoKeras (run)
	#   AutoML method Based on: 
	#      Jin, H., Song, Q. and Hu, X., 2019, July. Auto-keras: An efficient neural architecture search system. 
	#      In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & 
	#      Data Mining (pp. 1946-1956). ACM.
	#   Script adapted from:
	#      https://github.com/keras-team/autokeras

	## Define run folder path to store generated models and data
	run_folder = 'autokeras/regression/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	# Autokeras execution
	akc = AutoKerasRegression(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, verbosity=verbosity, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)
       
	final_model, transform_obj, results = akc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])

	print("#################################################################################################")
	print("##############################            RUNNING TPOT                ###########################")
	print("#################################################################################################")

	# TPOT (run)
	#   AutoML method Based on: 
	#      Olson, R.S., Bartley, N., Urbanowicz, R.J. and Moore, J.H., 2016, July. Evaluation of a tree-based 
	#      pipeline optimization tool for automating data science. In Proceedings of the Genetic and Evolutionary 
	#      Computation Conference 2016 (pp. 485-492). ACM.
	#   Script adapted from:
	#      https://github.com/EpistasisLab/tpot

	## Define run folder path to store generated models and data
	run_folder = 'tpot/regression/'

	# Refresh model folder
	if not os.path.isdir(model_folder + run_folder):
		os.mkdir(model_folder + run_folder)

	# Refresh outputs folder
	if not os.path.isdir(output_folder + run_folder):
		os.mkdir(output_folder + run_folder)

	tpc = TPOTRegression(data_path, model_folder + run_folder, output_folder + run_folder, max_runtime=max_runtime_minutes, num_folds=num_folds, sequence_type=sequence_type, verbosity=verbosity, population_size=population_size, num_generations=num_generations, input_col=input_col, target_col=target_col, pad_seqs=pad_seqs, augment_data=augment_data, dataset_robustness=dataset_robustness, run_interpretation = run_interpretation, interpret_params = interpret_params, run_design = run_design, design_params = design_params)

	final_model, transform_obj, results = tpc.run_system()

	# Create backup folder
	if do_backup:
	    backup_folder = './backup/runs/' +  datetime.now().strftime('%Y%m%d') + '/' + run_folder
	    ## Create folder to store model (if not existent)
	    if not os.path.isdir(backup_folder):
	        os.makedirs(backup_folder)

	    # Copy all contents to dated backup
	    copy_full_dir(model_folder + run_folder, backup_folder + model_folder[2:])
	    copy_full_dir(output_folder + run_folder, backup_folder + output_folder[2:])


def run_bioseqml(task, data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, do_auto_bin, bin_threshold, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params):
	data_path = data_folder + data_file 
     
	# Refresh model folder
	if not os.path.isdir(model_folder):
		os.mkdir(model_folder)

	# Refresh model folder
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	# Refresh model folder
	if not os.path.isdir(model_folder + 'deepswarm/'):
		os.mkdir(model_folder + 'deepswarm/')

	# Refresh model folder
	if not os.path.isdir(model_folder + 'autokeras/'):
		os.mkdir(model_folder + 'autokeras/')

	# Refresh model folder
	if not os.path.isdir(model_folder + 'tpot/'):
		os.mkdir(model_folder + 'tpot/')

	# Refresh model folder
	if not os.path.isdir(output_folder + 'deepswarm/'):
		os.mkdir(output_folder + 'deepswarm/')

	# Refresh model folder
	if not os.path.isdir(output_folder + 'autokeras/'):
		os.mkdir(output_folder + 'autokeras/')

	# Refresh model folder
	if not os.path.isdir(output_folder + 'tpot/'):
		os.mkdir(output_folder + 'tpot/')

	if task == 'binary_classification':
		print("#################################################################################################")
		print("#######################               RUNNING BINARY CLASSIFICATION            ##################")
		print("#################################################################################################")
		print('')
		run_bioseqml_binaryclass(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, do_auto_bin, bin_threshold, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params)

	elif task == 'multiclass_classification':
		print("#################################################################################################")
		print("#######################            RUNNING MULTICLASS CLASSIFICATION           ##################")
		print("#################################################################################################")
		print('')
		run_bioseqml_multiclass(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, do_auto_bin, bin_threshold, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params)
		
	elif task == 'regression':
		print("#################################################################################################")
		print("#######################               RUNNING REGRESSION                  #######################")
		print("#################################################################################################")
		print('')
		run_bioseqml_regression(data_folder, data_file, sequence_type, model_folder, output_folder, max_runtime_minutes, num_folds, verbosity, do_backup, input_col, target_col, pad_seqs, augment_data, dataset_robustness, num_final_epochs, yaml_params, num_generations, population_size, run_interpretation, interpret_params, run_design, design_params)

	else:
		print('Task is not valid. Please choose one of binary_classification, multiclass_classification, or regression.')