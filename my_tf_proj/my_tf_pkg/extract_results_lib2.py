import json
import os
import pdb

import numpy as np
import re

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
    return train_error, cv_error, test_error

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
    return train_error, cv_error, test_error

def get_most_recent_error(train_errors, cv_errors, test_errors):
    # get most recent error
    (train_error, cv_error, test_error) = train_errors[-1], cv_errors[-1], test_errors[-1]
    return (train_error, cv_error, test_error)

#

def get_list_errors(experiment_results,get_errors_from):
    # experiment_results : units->results
    print( 'get_list_errors2')
    list_units = []
    list_train_errors = []
    list_test_errors = []
    #print experiment_results
    for nb_units, results in experiment_results.items():
        #print 'nb_units ', nb_units
        #print 'results', results
        train_error, cv_error, test_error = get_errors_from(results)
        #print '--nb_units', nb_units
        #print 'train_error, cv_error, test_error ', train_error, cv_error, test_error
        list_units.append(nb_units)
        list_train_errors.append(train_error)
        list_test_errors.append(test_error)
    # sort based on first list
    print( len(list_train_errors))
    print( len(list_test_errors))
    #_, list_train_errors = zip(*sorted(zip(list_units, list_train_errors)))
    #list_units, list_test_errors = zip( *sorted( zip(list_units, list_test_errors)  ) )
    #
    _, list_train_errors = sort_and_pair_units_with_errors(list_units, list_train_errors)
    list_units, list_test_errors = sort_and_pair_units_with_errors(list_units, list_test_errors)
    return list_units, list_train_errors, list_test_errors

#

def get_results(dirpath, filename):
    '''
    Gets the results dictionary from a run.

    dirpath - the directory path to the task_experiment directory with all the different experiements for the current task.
    filename - one specific run (i.e. specific hyperparameter setting) for the current dirpath experiment_task
    '''
    path_to_json_file = dirpath+'/'+filename
    with open(path_to_json_file, 'r') as data_file:
        results = json.load(data_file)
    return results

def get_best_results_from_experiment(experiment_dirpath, list_runs_filenames, get_errors_from):
    '''
        Returns the best result structure for the current experiment from all the runs.

        list_runs_filenames = filenames list with all the specific simulation runs (i.e. specific hyper parameters for the current experiment_dirpath/task)
        get_errors_from = function pointer/handler
    '''
    best_cv_errors = float('inf')
    best_cv_filname = None
    results_best = None
    final_train_errors = []
    final_cv_errors = []
    final_test_errors = []
    for run_filename in list_runs_filenames:
        # if current run=filenmae is a json struct then it has the results
        if 'json' in run_filename:
            print('run_filename', run_filename)
            with open(experiment_dirpath+'/'+run_filename, 'r') as data_file:
                results_current_run = json.load(data_file)
            #results_current_run = get_results(experiment_dirpath, run_filename)

            train_error, cv_error, test_error = get_errors_from(results_current_run)
            final_train_errors.append(train_error)
            final_cv_errors.append(cv_error)
            final_test_errors.append(test_error)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_cv_filname = run_filename
                results_best = results_current_run
    return results_best, best_cv_filname, final_train_errors, final_cv_errors, final_test_errors

def get_results_for_experiments(path_to_all_experiments_for_task, get_errors_from, verbose=True):
    '''

    example:
    path_to_all_experiments_for_task = ../../om_mnist/task_August_7_NN1_xavier_momentum
    '''
    experiment_results = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        #dirpath = om_task_data_set/august_NN1_xavier/
        if (dirpath != path_to_all_experiments_for_task): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            results_best, best_filename, final_train_errors, final_cv_errors, final_test_errors = get_best_results_from_experiment(
                experiment_dirpath=experiment_dirpath,
                list_runs_filenames=potential_runs_filenames,
                get_errors_from=get_errors_from)


    return experiment_results

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
