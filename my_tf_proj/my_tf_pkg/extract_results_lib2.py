import json
import os
import pdb

import numpy as np
import re

import namespaces as ns

##

def get_errors_based_on_train_error(results):
    '''
        Gets the train,test errors based on the minimizer of the train error for the current simulation run.
        The train error is the min train error and test error is the corresponding test error of the model
        with the smallest train error.
    '''
    # get error lists
    (train_errors, cv_errors, test_errors) = (results['train_errors'], results['cv_errors'], results['test_errors'])
    min_train_index = np.argmin(train_errors)
    (train_error, cv_error, test_error) = train_errors[min_train_index], cv_errors[min_train_index], test_errors[min_train_index]
    return train_error, train_error, cv_error, test_error

def get_errors_based_on_validation_error(results):
    '''
        Gets the train,validation,test errors based on the minimizer of the validation error dor the current simulation run.
        The validation error is the min validation error and test error is the corresponding test error of the model
        with the smallest validation error. Similarly, the train error is the train error of the smallest validation error.
    '''
    # get error lists
    (train_errors, cv_errors, test_errors) = (results['train_errors'], results['cv_errors'], results['test_errors'])
    min_cv_index = np.argmin(cv_errors)
    (train_error, cv_error, test_error) = train_errors[min_cv_index], cv_errors[min_cv_index], test_errors[min_cv_index]
    return cv_error, train_error, cv_error, test_error

# def get_most_recent_error(train_errors, cv_errors, test_errors):
#     # get most recent error
#     (train_error, cv_error, test_error) = train_errors[-1], cv_errors[-1], test_errors[-1]
#     return train_error, cv_error, test_error

#

def sort_and_pair_units_with_errors(list_units,list_test_errors):
    '''
        Gets a list of units and a list of error that correspond to each other and sorts it into two seperate lists.
        Note that the error[i] must correspond to the error given by unit[i] for the algorithm to work.

        Ex: units = [12,5,20], errors = [5.3,20.2,0.1] => [5,12,20],[20.2,5.3,0.1]

        implements: list_units, list_test_errors = zip( *sorted( zip(list_units, list_test_errors)  ) )
        http://stackoverflow.com/questions/39629253/what-does-the-code-zip-sorted-zipunits-errors-do
    '''
    #list_units, list_test_errors = zip( *sorted( zip(list_units, list_test_errors)  ) )
    units_error_pair_list = zip(list_units, list_test_errors) # [...,(unit, error),...]
    # sort by first entry (break ties by second, essentially sorts tuples a number is sorted)
    sort_by_units = sorted(units_error_pair_list) # [..,(units, error),...] but units_i < units_j (i.e. sorted by unit number)
    # collect the units in one list and the errors in another (note they are sorted), note the * is needed to have zip not receive a single list
    list_units, list_test_errors = zip(*sort_by_units) # [units, ..., units] , [error, ..., error]
    return list_units, list_test_errors

#

def get_results_for_experiments(path_to_all_experiments_for_task,decider,verbose=True):
    '''

    example:
    path_to_all_experiments_for_task = ../../om_mnist/task_August_7_NN1_xavier_momentum

    ../../om_mnist/expts_task_August_7_NN/NN1_xavier/json_*
                                          NN2_xavier/json_*
                                          NN3_xavier/json_*
    '''
    expts_best_results = {} #maps units -> to corresponding best data
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        #dirpath = om_task_data_set/august_NN1_xavier/NN1_xavier
        #dirnames = _ (essentially empty for what we care)
        #filenames = [file conents of current dirpath]
        if (dirpath != path_to_all_experiments_for_task): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            print('dirpath ' , dirpath)
            best_data = get_best_results_from_experiment(experiment_dirpath=dirpath,list_runs_filenames=filenames,decider=decider)
            #
            nb_units = best_data.results_best['dims'][1]
            expts_best_results[nb_units] = best_data
    #print(expts_best_results)
    return expts_best_results

def get_best_results_from_experiment(experiment_dirpath,list_runs_filenames,decider):
    '''
    '''
    #the error that we make decision based on (usually train or validation, train for ERM, validation for CV)
    #best_decider_error = float('inf')
    best_data = ns.Namespace(best_decider_error=float('inf'))
    for run_filename in list_runs_filenames:
        if 'json' in run_filename: # if current run=filenmae is a json struct then it has the results
            #print('run_filename', run_filename)
            with open(experiment_dirpath+'/'+run_filename, 'r') as data_file:
                results_current_run = json.load(data_file)
            decider_error, train_error, cv_error, test_error = decider.get_errors_from(results_current_run)
            if decider_error < best_data.best_decider_error:
                update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error)
    return best_data

def update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error):
    best_data.best_decider_error = decider_error
    best_data.best_decider_filname = run_filename
    best_data.results_best = results_current_run

    best_data.best_train_error = train_error
    best_data.best_databest_cv_error = cv_error
    best_data.best_test_error = test_error
