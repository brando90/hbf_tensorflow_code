import json
import os
import pdb

import numpy as np
import re

import six

import namespaces as ns

import pdb
import pandas as pd

##

def get_errors_based_on_train_error(errors_df):
    '''
        Gets the train,test errors based on the minimizer of the train error for the current simulation run.
        The train error is the min train error and test error is the corresponding test error of the model
        with the smallest train error.
    '''
    # get error lists
    train_errors, cv_errors, test_errors = (errors_df['train_errors'], errors_df['cv_errors'], errors_df['test_errors'])
    min_train_index = np.nanargmin(train_errors) #Return the indices of the minimum values in the specified axis ignoring NaNs
    (train_error, cv_error, test_error) = train_errors[min_train_index], cv_errors[min_train_index], test_errors[min_train_index]
    return train_error, train_error, cv_error, test_error

def get_errors_based_on_validation_error(errors_df):
    '''
        Gets the train,validation,test errors based on the minimizer of the validation error dor the current simulation run.
        The validation error is the min validation error and test error is the corresponding test error of the model
        with the smallest validation error. Similarly, the train error is the train error of the smallest validation error.
    '''
    # get error lists
    (train_errors, cv_errors, test_errors) = (errors_df['train_errors'], errors_df['cv_errors'], errors_df['test_errors'])
    min_cv_index = np.nanargmin(cv_errors) #Return the indices of the minimum values in the specified axis ignoring NaNs
    (train_error, cv_error, test_error) = train_errors[min_cv_index], cv_errors[min_cv_index], test_errors[min_cv_index]
    return cv_error, train_error, cv_error, test_error

#

def get_best_results_for_experiments_csv(path_to_all_experiments_for_task,decider,verbose=True,mdl_complexity_criteria='nb_units'):
    '''
    Given a path to all the experiments for a specific task, goes through each individual folder for each experiment for each different
    model tested and extracts the lowest error according to the decider. For example, the if the decider is set to choose based on train
    error then it will collect the model corresponding to the lowest train error (and also collect the validation,test error and hyper parameters for that model).

    path_to_all_experiments_for_task = path to ../../TASK/EXPT_NAME
    decider = namespace holding the appropriate function handler/pointer named get_errors_from (e.g. get_errors_based_on_train_error).
    So decider must be able to call decider.get_errors_from(run)
    mdl_complexity_criteria = the key or criteria to choose the mdl complexity.
    Goal is to map {mdl_complexity_key:error}.
    Example of mdl_complexity_criteria='nb_units' or 'nb_params'

    example:
    path_to_all_experiments_for_task = ../../om_mnist/task_August_7_NN1_xavier_momentum

    ../../om_mnist/expts_task_August_7_NN/expt_NN1_xavier/run_json_* lots of these
                                          expt_NN2_xavier/run_json_*
                                          expt_NN3_xavier/run_json_*

    for each model/expt NN select best hyper params based on decider.
    '''
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        #dirpath = om_task_data_set/august_NN1_xavier/NN1_xavier
        #dirnames = _ (essentially empty for what we care)
        #filenames = [file conents of current dirpath]
        #print('dirpath: %s, dirnames: %s, filenames: %s'%(dirpath, dirnames, filenames))
        if (dirpath != path_to_all_experiments_for_task): # if current dirpath is a valid experiment and not . (itself)
            #print('>>>dirpath: %s, dirnames: %s, filenames: %s'%(dirpath, dirnames, filenames))
            best_results = _get_best_results_obj_from_current_experiment(model_path=dirpath,list_runs_filenames=filenames,decider=decider)
            mdl_complexity_key = get_key_for_mdl_complexity(mdl_complexity_criteria,best_data)

            # check if there are repeated runs/simulations results for this dirpath, choose the better of the two
            print('%s: '%(mdl_complexity_criteria), mdl_complexity_key )
            if mdl_complexity_key in expts_best_results:
                prev_data = expts_best_results[mdl_complexity_key]
                if best_data.best_decider_error < prev_data.best_decider_error:
                    expts_best_results[mdl_complexity_key] = best_data
                    #expts_best_results[nb_params] = best_data
            else:
                expts_best_results[mdl_complexity_key] = best_data
                #expts_best_results[nb_params] = best_data
    print(expts_best_results)
    return expts_best_results

def get_best_results_for_model(model_path,list_runs_filenames,decider):
    '''
    Given a specific model path, it goes through all the file runs inside (i.e. csv and hp.json files) and gets the best models according to
    decider (e.g. according to train error etc).

    experiment_dirpath = path to experiments. (e.g ../../om_mnist/expts_task_August_7_NN/expt_NN1_xavier)
    list_runs_filenames = the list of run results for each hyperparam for the current experiment (ideally json files with the results)
    '''
    #the error that we make decision based on (usually train or validation, train for ERM, validation for CV)
    best_data = ns.Namespace(best_decider_error=float('inf'))
    for run_filename in list_runs_filenames:
        if 'csv' in run_filename:
            errors_df = pd.read_csv(experiment_dirpath+'/'+run_filename) # pandas data frame
            decider_error, train_error, cv_error, test_error = decider.get_errors_from(errors_df)
            if decider_error < best_data.best_decider_error:
                _update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error)

    return best_data

def _update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error):
    '''
    Updates the structure best_data with the new data for the current model.
    This should only be called when decider_error < best_data.best_decider_error.
    '''
    with open(experiment_dirpath+'/'+run_filename, 'r') as data_file:
        results_current_run = json.load(data_file)
    #
    best_data.best_decider_error = decider_error
    best_data.best_decider_filname = run_filename
    best_data.results_best = results_current_run

    best_data.best_train_error = train_error
    best_data.best_cv_error = cv_error
    best_data.best_test_error = test_error

if __name__ == '__main__':
    path_to_all_experiments_for_task = '../../../simulation_results_scripts/om_f_256D_L8_ppt_1/TMP_hp_test/'
    get_errors_from = ''
    decider = ns.Namespace(get_errors_from=get_errors_from)
    # run (unit) test
    get_best_results_for_experiments_csv(path_to_all_experiments_for_task,decider,verbose=True,mdl_complexity_criteria='nb_units')
