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
    min_train_index = np.nanargmin(train_errors) #Return the indices of the minimum values in the specified axis ignoring NaNs
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
    min_cv_index = np.nanargmin(cv_errors) #Return the indices of the minimum values in the specified axis ignoring NaNs
    (train_error, cv_error, test_error) = train_errors[min_cv_index], cv_errors[min_cv_index], test_errors[min_cv_index]
    return cv_error, train_error, cv_error, test_error

def get_most_recent_error(train_errors, cv_errors, test_errors):
    # get most recent error
    (train_error, cv_error, test_error) = train_errors[-1], cv_errors[-1], test_errors[-1]
    return (train_error, cv_error, test_error)

#

def get_best_results_for_experiments(path_to_all_experiments_for_task,decider,verbose=True):
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
            best_data = _get_best_results_obj_from_current_experiment(experiment_dirpath=dirpath,list_runs_filenames=filenames,decider=decider)
            #
            #nb_units = best_data.results_best['dims'][1]
            #pdb.set_trace()
            #print(best_data)
            nb_units = best_data.results_best['arg_dict']['dims'][1] if not 'dims' in best_data.results_best else best_data.results_best['dims'][1]
            del best_data['results_best']
            # check if there are repeated runs/simulations results for this dirpath, choose the better of the two
            print( nb_units )
            if nb_units in expts_best_results:
                prev_data = expts_best_results[nb_units]
                if best_data.best_decider_error < prev_data.best_decider_error:
                    expts_best_results[nb_units] = best_data
            else:
                expts_best_results[nb_units] = best_data
    #print(expts_best_results)
    return expts_best_results

def _get_best_results_obj_from_current_experiment(experiment_dirpath,list_runs_filenames,decider):
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
                _update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error)
    return best_data

def _update(best_data,decider_error,run_filename,results_current_run, train_error,cv_error,test_error):
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

#

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

def _get_results(dirpath, filename):
    '''
    Gets the results dictionary from a run.

    dirpath - the directory path to the task_experiment directory with all the different experiements for the current task.
    filename - one specific run (i.e. specific hyperparameter setting) for the current dirpath experiment_task
    '''
    if 'json' in filename: # if current run=filenmae is a json struct then it has the results
        with open(experiment_dirpath+'/'+filename, 'r') as data_file:
            results_current_run = json.load(data_file)
    return results_current_run

def get_all_simulation_results(path_to_all_experiments_for_task,decider,verbose=True):
    '''
    Given a path to all the experiments for a specific task, goes through each individual folder for each model
    and returns ALL results a specific model.

    If a model has more than one dirpath, we return the results for *both*.
    Also, for a specific model we might have e.g. 200 runs. Each run has lots of iterations.

    returns { model: [best_over_run_i]^200_i } = { model: [ min runs_j ]^200_i }
    where best_over_run_i = min over all the iterions
    '''
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        #dirpath = om_task_data_set/august_NN1_xavier/NN1_xavier
        #dirnames = _ (essentially empty for what we care)
        #filenames = [file conents of current dirpath]
        if (dirpath != path_to_all_experiments_for_task) and (not 'mdls' in dirpath): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            print('dirpath ' , dirpath)
            # get all results for all runs for current model/dirpath
            all_results_for_current_mdl = []
            for filename in filenames: #for run in all_runs
                if 'json' in filename: # if current run=filenmae is a json struct then it has the results
                    with open(dirpath+'/'+filename, 'r') as data_file:
                        results_current_run = json.load(data_file)
                    all_results_for_current_mdl.append(results_current_run)
            #
            nb_units = _get_nb_units(all_results_for_current_mdl[0])
            # check if there are repeated runs/simulations results for this dirpath, simply remember all results for all experiments
            if nb_units in expts_best_results:
                expts_best_results[nb_units] = expts_best_results[nb_units] + all_results_for_current_mdl
            else:
                expts_best_results[nb_units] = all_results_for_current_mdl
    return expts_best_results

def _get_nb_units(results):
    '''
    gets the number of units for these results
    '''
    return results['arg_dict']['dims'][1] if not 'dims' in results else results['dims'][1]

#

def get_mean_std(all_results, decider):
    '''
        Gets the mean and std errors for results.

        all_results = dict with all results. Each nb_units (key) maps to the results for all runs, usually a run corressponds
        to a specific hyperparam setting or repeated run for SGD/initialization
        e.g. {units:[results1,...,results200]}
        decider = namespace holding the appropriate function handler/pointer named get_errors_from (e.g. get_errors_based_on_train_error).
        So decider must be able to call decider.get_errors_from(run)
    '''
    units = []
    means_decider = []
    stds_decider = []
    means_test = []
    stds_test = []
    for nb_units, results_for_runs in six.iteritems(all_results):
        # gets the mean,std over the runs
        decider_mean, decider_std, test_mean, test_std = _get_mean_std_from_runs(results_for_runs,decider)
        print('decider_mean', decider_mean)
        # update
        units.append(nb_units)
        #
        means_decider.append(decider_mean)
        stds_decider.append(decider_std)
        #
        means_test.append(test_mean)
        stds_test.append(test_std)
    #sort and pair up units with errors
    sorted_units, sorted_decider_mean = sort_and_pair_units_with_errors(list_units=units,list_errors=means_decider)
    sorted_units, sorted_decider_std = sort_and_pair_units_with_errors(list_units=units,list_errors=stds_decider)
    sorted_units, sorted_test_mean = sort_and_pair_units_with_errors(list_units=units,list_errors=means_test)
    sorted_units, sorted_test_std = sort_and_pair_units_with_errors(list_units=units,list_errors=stds_test)
    return sorted_units, sorted_decider_mean, sorted_decider_std, sorted_test_mean, sorted_test_std

def _get_mean_std_from_runs(results_for_runs,decider):
    '''
    For a collection of runs (usually from HPs) return the average and std of the decider error and test error.
    Usually decider error will be validation or train error (which we then also get the average test error)

    results_for_runs = array with all results for runs (each run usually corresponds to a speicfic HP) for a specific model
        e.g. [result1, ..., result200]
    decider = namespace holding the appropriate function handler/pointer named get_errors_from (e.g. get_errors_based_on_train_error).
    So decider must be able to call decider.get_errors_from(run)
    '''
    decider_errors_for_runs = [] #
    #train_errors_for_runs = []
    #cv_errors_for_runs = []
    test_errors_for_runs = [] #
    for current_result in results_for_runs:
        decider_error, train_error, cv_error, test_error = decider.get_errors_from(current_result)
        print('decider_error ', decider_error)
        #
        # if np.isnan( decider_error ):
        #     pdb.set_trace()
        decider_errors_for_runs.append(decider_error)
        #train_errors_for_runs.append(train_error)
        #cv_errors_for_runs.append(cv_error)
        test_errors_for_runs.append(test_error)
    decider_mean, decider_std = np.mean(decider_errors_for_runs), np.std(decider_errors_for_runs)
    test_mean, test_std = np.mean(test_errors_for_runs), np.std(test_errors_for_runs)
    #pdb.set_trace()
    return decider_mean, decider_std, test_mean, test_std

#

def get_duration_of_experiments(all_results, duration_type='minutes'):
    '''
    For a collection of runs (usually from HPs) return the average and std of the duration of the runs
    i.e. how long it takes to complete one simulation/run

    all_results = dict with all results. Each nb_units (key) maps to the results for all runs, usually a run corressponds
    to a specific hyperparam setting or repeated run for SGD/initialization
    e.g. {units:[results1,...,results200]}
    duration_type = units for duration e.g. 'second', 'minutes', 'hours'
    '''
    units = []
    duration_means = []
    duration_stds = []
    for nb_units, results_for_runs in six.iteritems(all_results):
        duration_mean, duration_std = _get_duration_mean_std_from_runs(results_for_runs, duration_type)
        units.append(nb_units)
        duration_means.append(duration_mean)
        duration_stds.append(duration_std)
    #sort and pair up units with errors
    sorted_units, sorted_duration_means = sort_and_pair_units_with_errors(list_units=units,list_errors=duration_means)
    sorted_units, sorted_duration_stds = sort_and_pair_units_with_errors(list_units=units,list_errors=duration_stds)
    return sorted_units, sorted_duration_means, sorted_duration_stds

def _get_duration_mean_std_from_runs(results_for_runs, duration_type):
    '''
    For a collection of runs (usually from HPs) return the average and std duration.

    results_for_runs = array with all results for runs (each run usually corresponds to a speicfic HP) for a specific model
        e.g. [result1, ..., result200]
    duration_type = units for duration e.g. 'second', 'minutes', 'hours'
    '''
    durations = []
    for current_result in results_for_runs:
        duration = _get_durations(current_result,duration_type)
        durations.append(duration)
    duration_mean, duration_std = np.mean(durations), np.std(durations)
    return duration_mean, duration_std

def _get_durations(result, duration_type):
    '''
    Gets the duration for current result given.

    result = the standard resultd dictionary mapping to info about run/simulation.
    duration_type = units for duration e.g. 'second', 'minutes', 'hours'
    '''
    duration = result[duration_type] #result['seconds'], result['minutes'], result['hours']
    return duration
