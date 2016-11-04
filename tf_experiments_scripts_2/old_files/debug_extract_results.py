import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls
import my_tf_pkg as mtf
import namespaces as ns
##

def display_results_task_f_4D_conv_2nd():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a
    #
    get_errors_from = mtf.get_errors_based_on_train_error
    #get_errors_from = mtf.get_errors_based_on_validation_error
    decider = ns.Namespace(get_errors_from=get_errors_from)
    #
    task_name = 'task_f_4D_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)

    path_to_experiments = '../../%s/task_September_23_NN_xavier_softplus'%experiment_name
    expts_best_results = mtf.get_results_for_experiments(path_to_experiments,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    krls.plot_values(nb_params_shallow,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Shallow NN',markersize=3,colour='b')
    krls.plot_values(nb_params_shallow,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test error for Shallow NN',markersize=3,colour='b')

    path_to_experiments = '../../%s/task_September_23_BTHL_xavier_softplus'%experiment_name
    expts_best_results = mtf.get_results_for_experiments(path_to_experiments,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    krls.plot_values(nb_params_bt,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    krls.plot_values(nb_params_bt,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    #
    plt.legend()
    plt.show()

def debug(path_to_all_experiments_for_task,decider,verbose=True):
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        if (dirpath != path_to_all_experiments_for_task) and (not 'mdls' in dirpath): # if current dirpath is a valid experiment and not . (itself)
            #print('=> potential_runs_filenames: ', potential_runs_filenames)
            print('dirpath ' , dirpath)
            #nb_units = best_data.results_best['dims'][1]
            #best_data = ns.Namespace(best_decider_error=float('inf'))
            #errors_for_dirpath = []
            for run_filename in filenames:
                if 'json' in run_filename: # if current run=filenmae is a json struct then it has the results
                    #print('run_filename', run_filename)
                    with open(dirpath+'/'+run_filename, 'r') as data_file:
                        results_current_run = json.load(data_file)
                    #decider_error, train_error, cv_error, test_error = decider.get_errors_from(results_current_run)
                    #errors_for_dirpath.append(train_error)
            results_best = results_current_run
            nb_units = results_best['arg_dict']['dims'][1] if not 'dims' in best_data.results_best else results_best['dims'][1]
            #del best_data['results_best']
            expts_best_results[nb_units] = np.min(errors_for_dirpath)
    #print(expts_best_results)
    return expts_best_results

def main():
    get_errors_from = mtf.get_errors_based_on_train_error
    #get_errors_from = mtf.get_errors_based_on_validation_error
    decider = ns.Namespace(get_errors_from=get_errors_from)
    task_name = 'task_f_4D_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)
    path_to_experiments = '../../%s/task_September_23_NN_xavier_softplus'%experiment_name
    #
    debug(path_to_all_experiments_for_task=path_to_experiments,decider=decider,verbose=True)

if __name__ == '__main__':
    main()
