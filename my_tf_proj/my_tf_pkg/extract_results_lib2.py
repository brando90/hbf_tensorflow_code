import json
import os
import pdb

import numpy as np
import re

import six

import namespaces as ns

import pdb

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

def get_most_recent_error(train_errors, cv_errors, test_errors):
    # get most recent error
    (train_error, cv_error, test_error) = train_errors[-1], cv_errors[-1], test_errors[-1]
    return (train_error, cv_error, test_error)

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

#

def get_results_for_experiments(path_to_all_experiments_for_task,decider,verbose=True):
    '''
    Given a path to all the experiments for a specific task, goes through each individual folder for each experiment for each different
    model tested and extracts the lowest error according to the decider. For example, the if the decider is set to choose based on train
    error then it will collect the model corresponding to the lowest train error (and also collect the validation,test error and hyper parameters for that model).

    path_to_all_experiments_for_task = path to ../../TASK/EXPT_NAME
    decider = namespace holding the appropriate function handler/pointer named get_errors_from (e.g. get_errors_based_on_train_error).
    So decider must be able to call decider.get_errors_from(run)

    example:
    path_to_all_experiments_for_task = ../../om_mnist/task_August_7_NN1_xavier_momentum

    ../../om_mnist/expts_task_August_7_NN/expt_NN1_xavier/run_json_* lots of these
                                          expt_NN2_xavier/run_json_*
                                          expt_NN3_xavier/run_json_*

    for each model/expt NNi select best hyper params based on decider.
    '''
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        #dirpath = om_task_data_set/august_NN1_xavier/NN1_xavier
        #dirnames = _ (essentially empty for what we care)
        #filenames = [file conents of current dirpath]
        if (dirpath != path_to_all_experiments_for_task) and (not 'mdls' in dirpath): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            print('dirpath ' , dirpath)
            best_data = get_best_results_from_experiment(experiment_dirpath=dirpath,list_runs_filenames=filenames,decider=decider)
            #
            #nb_units = best_data.results_best['dims'][1]
            nb_units = best_data.results_best['arg_dict']['dims'][1] if not 'dims' in best_data.results_best else results_best['dims'][1]
            del best_data['results_best']
            expts_best_results[nb_units] = best_data
    #print(expts_best_results)
    return expts_best_results

def get_best_results_from_experiment(experiment_dirpath,list_runs_filenames,decider):
    '''
    Given a specific experiment path, it goes through all the file runs inside (i.e. json files) and gets the best models according to
    decider (e.g. according to train error etc).

    experiment_dirpath = path to experiments. (e.g ../../om_mnist/expts_task_August_7_NN/expt_NN1_xavier)
    list_runs_filenames = the list of run results for each hyperparam for the current experiment (ideally json files with the results)
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
    '''
    Updates the structure best_data with the new data for the current model.
    This should only be called when decider_error < best_data.best_decider_error.
    '''
    best_data.best_decider_error = decider_error
    best_data.best_decider_filname = run_filename
    best_data.results_best = results_current_run

    best_data.best_train_error = train_error
    best_data.best_cv_error = cv_error
    best_data.best_test_error = test_error

def get_errors_for_display(expts_best_results):
    '''
        Extracts a list of units and errors that corressponds to each other an they are sorted by the units.

        So the idea is to get unordered dict to be
    '''
    units = []
    train_errors = []
    validation_errors = []
    test_errors = []

    #best_filenames = []
    for key in six.iterkeys(expts_best_results):
        best_data = expts_best_results[key]

        units.append(key)
        train_errors.append( best_data.best_train_error )
        validation_errors.append( best_data.best_cv_error )
        test_errors.append( best_data.best_test_error )

        #best_filenames.append( best_data.best_decider_filname )
    # sort and pair up units with errors
    sorted_units, sorted_train_errors = sort_and_pair_units_with_errors(list_units=units,list_errors=train_errors)
    sorted_units, sorted_validation_errors = sort_and_pair_units_with_errors(list_units=units,list_errors=validation_errors)
    sorted_units, sorted_test_errors = sort_and_pair_units_with_errors(list_units=units,list_errors=test_errors)
    return sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors

def sort_and_pair_units_with_errors(list_units,list_errors):
    '''
        Gets a list of units and a list of error that correspond to each other and sorts it into two seperate lists.
        Note that the error[i] must correspond to the error given by unit[i] for the algorithm to work.

        Ex: units = [12,5,20], errors = [5.3,20.2,0.1] => [5,12,20],[20.2,5.3,0.1]

        implements: list_units, list_test_errors = zip( *sorted( zip(list_units, list_test_errors)  ) )
        http://stackoverflow.com/questions/39629253/what-does-the-code-zip-sorted-zipunits-errors-do
    '''
    #list_units, list_test_errors = zip( *sorted( zip(list_units, list_test_errors)  ) )
    units_error_pair_list = zip(list_units, list_errors) # [...,(unit, error),...]
    # sort by first entry (break ties by second, essentially sorts tuples a number is sorted)
    sort_by_units = sorted(units_error_pair_list) # [..,(units, error),...] but units_i < units_j (i.e. sorted by unit number)
    # collect the units in one list and the errors in another (note they are sorted), note the * is needed to have zip not receive a single list
    list_units_out, list_errors_out = zip(*sort_by_units) # [units, ..., units] , [error, ..., error]
    return list(list_units_out), list(list_errors_out)

#

def debug(experiment_dirpath,list_runs_filenames,decider):
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        if (dirpath != path_to_all_experiments_for_task) and (not 'mdls' in dirpath): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            print('dirpath ' , dirpath)
            #nb_units = best_data.results_best['dims'][1]
            #best_data = ns.Namespace(best_decider_error=float('inf'))
            errors_for_dirpath = []
            for run_filename in filenames:
                if 'json' in run_filename: # if current run=filenmae is a json struct then it has the results
                    #print('run_filename', run_filename)
                    with open(dirpath+'/'+run_filename, 'r') as data_file:
                        results_current_run = json.load(data_file)
                    decider_error, train_error, cv_error, test_error = decider.get_errors_from(results_current_run)
                    errors_for_dirpath.append(train_error)
                    # if decider_error < best_data.best_decider_error:
                    #     update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error)
            #
            nb_units = best_data.results_best['arg_dict']['dims'][1] if not 'dims' in best_data.results_best else results_best['dims'][1]
            #del best_data['results_best']
            expts_best_results[nb_units] = np.min(errors_for_dirpath)
    #print(expts_best_results)
    return expts_best_results
