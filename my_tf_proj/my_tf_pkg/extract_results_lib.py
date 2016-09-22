import json
import os
import pdb

import numpy as np
import re

##

def get_errors_based_on_train_error(results):
    '''
        Gets the train,test errors based on the minimizer of the train error.
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
        Gets the train,validation,test errors based on the minimizer of the validation error.
        The validation error is the min validation error and test error is the corresponding test error of the model
        with the smallest validation error. Similarly, the train error is the train error of the smallest validation error.
    '''
    # get error lists
    (train_errors, cv_errors, test_errors) = (results['train_errors'], results['cv_errors'], results['test_errors'])
    # get most recent error
    #(train_error, cv_error, test_error) = train_errors[-1], cv_errors[-1], test_errors[-1]
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
    train_error, cv_error, test_error = (None, None, None)
    results = None
    path_to_json_file = dirpath+'/'+filename
    #print 'path_to_json_file', path_to_json_file
    with open(path_to_json_file, 'r') as data_file:
        results = json.load(data_file)
    return results

def get_best_results_from_experiment(experiment_dirpath, list_runs_filenames, get_errors_from):
    '''
        Returns the best result structure for the current experiment from all the runs.

        list_runs_filenames = filenames list with potential runs
    '''
    best_cv_errors = float('inf')
    best_cv_filname = None
    results_best = None
    final_train_errors = []
    final_cv_errors = []
    final_test_errors = []
    for potential_run_filename in list_runs_filenames:
        # if current run=filenmae is a json struct then it has the results
        if 'json' in potential_run_filename:
            #print 'potential_run_filename', potential_run_filename
            run_filename = potential_run_filename
            results_current_run = get_results(experiment_dirpath, run_filename)
            train_error, cv_error, test_error = get_errors_from(results_current_run)
            final_train_errors.append(train_error)
            final_cv_errors.append(cv_error)
            final_test_errors.append(test_error)
            if cv_error < best_cv_errors:
                best_cv_errors = cv_error
                best_cv_filname = run_filename
                results_best = results_current_run
    return results_best, best_cv_filname, final_train_errors, final_cv_errors, final_test_errors

def get_results_for_experiments(path_to_experiments, get_errors_from, verbose=True, split_string='_jHBF[\d]*_|_jrun_HBF[\d]*_|_jNN[\d]*_'):
    '''
        Returns a dictionary containing the best results for each experiment
    '''
    print( '-----get_results_for_experiments')
    print( path_to_experiments )
    print( os.path.isdir(path_to_experiments) )
    #print os.listdir(path_to_experiments)
    experiment_results = {} # maps units -> results_best_mdl e.g {'4':{'results_best_mdl':results_best_mdl}}
    #print os.listdir(path_to_experiments)
    #print len(os.walk(path_to_experiments).next())
    for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
        print('experiment_dir', experiment_dir)
        #print 'potential_runs', potential_runs
        #print 'len(experiment_dir)', len(experiment_dir)
        #print 'len(potential_runs)', len(potential_runs)
        #print (experiment_dir != path_to_experiments)
        #print '> path_to_experiments: ',path_to_experiments
        #print '> experiment_dir: ',experiment_dir
        # if current dirpath is a valid experiment and not . (itself)
        if (experiment_dir != path_to_experiments):
            #print '=> experiment_dir: ', experiment_dir
            #print '=> potential_runs: ', potential_runs
            results_best, best_filename, final_train_errors, final_cv_errors, final_test_errors = get_best_results_from_experiment(experiment_dirpath=experiment_dir,list_runs_filenames=potential_runs,get_errors_from=get_errors_from)
            if results_best == None:
                continue
            if not 'dims' in results_best:
                nb_units = results_best['arg_dict']['dims'][1]
            else:
                nb_units = results_best['dims'][1]
            #(left, right) = experiment_dir.split('jHBF1_')
            #(left, right) = re.split('_jHBF[\d]*_',experiment_dir)
            # print('====> split_string: ', split_string)
            # print( '=====> SPLIT: ', re.split(split_string,experiment_dir))
            # split_res = re.split(split_string,experiment_dir)
            # print( '=====> split_res: ', split_res)
            # (left, right)  = split_res
            #pdb.set_trace()
            if verbose:
                print( '--')
                #print( right[0])
                print( 'experiment_dir ', experiment_dir)
                print( 'potential_runs ', len(potential_runs))
                print( 'type(potential_runs)', type(potential_runs))
                print( 'nb_units ', nb_units)
                print( 'best_filename ', best_filename)
            experiment_results[nb_units] = results_best
            experiment_results[nb_units]['final_train_errors'] = final_train_errors
            experiment_results[nb_units]['final_cv_errors'] = final_cv_errors
            experiment_results[nb_units]['final_test_errors'] = final_test_errors
    return experiment_results

def get_error_stats(experiment_results):
    '''
        Inserts (mutates) the dictionary results with mean std of errors.
    '''
    mean_train_errors = []
    mean_cv_errors = []
    mean_test_errors = []
    #
    std_train_errors = []
    std_cv_errors = []
    std_test_errors = []
    for nb_units in experiment_results.iterkeys():
        final_train_errors = experiment_results[nb_units]['final_train_errors']
        final_cv_errors = experiment_results[nb_units]['final_cv_errors']
        final_test_errors = experiment_results[nb_units]['final_test_errors']
        #
        mean_train_error = np.mean(final_train_errors)
        mean_cv_error = np.mean(final_cv_errors)
        mean_test_error = np.mean(final_test_errors)
        # experiment_results[nb_units]['mean_train_error'] = mean_train_error
        # experiment_results[nb_units]['mean_cv_error'] = mean_cv_error
        # experiment_results[nb_units]['mean_test_error'] = mean_test_error
        mean_train_errors.append(mean_train_error)
        mean_cv_errors.append(mean_cv_error)
        mean_test_errors.append(mean_test_error)
        #
        std_train_error = np.std(final_train_errors)
        std_cv_error = np.std(final_cv_errors)
        std_test_error = np.std(final_test_errors)
        # experiment_results[nb_units]['std_train_error'] = std_train_error
        # experiment_results[nb_units]['std_cv_error'] = std_cv_error
        # experiment_results[nb_units]['std_test_error'] = std_test_error
        std_train_errors.append(std_train_error)
        std_cv_errors.append(std_cv_error)
        std_test_errors.append(std_test_error)
    return mean_train_errors, std_train_errors, mean_test_errors, std_test_errors

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
