#print('here')
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

def shallow_vs_deep():
    get_errors_from = mtf.get_errors_based_on_train_error
    #get_errors_from = mtf.get_errors_based_on_validation_error
    decider = ns.Namespace(get_errors_from=get_errors_from)
    #
    task_name = 'f_256D_L8_ppt_1'
    #
    experiment_name = mtf.get_experiment_folder(task_name)
    print(experiment_name)

    transform = ''
    #transform = 'log'
    #plt.subplot(111)
    #axes = plt.gca()
    #axes.set_ylim(bottom=-0.0002,top=0.005)
    #axes.set_ylim(bottom=-0.00002,top=0.00053)

    path_to_experiments_NN = '../../simulation_results_scripts/%s/task_Dec_6_NN_256D_Adam_xavier_relu_N60000'%experiment_name
    path_to_experiments_NN = '../../simulation_results_scripts/%s/task_Jan_19_NN_256D_Adam_xavier_relu_N60000'%experiment_name
    path_to_experiments_NN = '../../simulation_results_scripts/%s/task_Jan_19_NN_256D_Adam_xavier_relu_N60000_original_setup'%experiment_name
    #mtf.combine_errors_and_hps_to_one_json_file(path_to_experiments_NN,verbose=True,overwrite_old=True)
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_NN,decider,verbose=False,mdl_complexity_criteria='nb_params',json_string='json')
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    nb_params_shallow = [ nb_units for nb_units in sorted_units ]
    #
    nb_params_shallow = np.array(nb_params_shallow)
    sorted_train_errors = np.array(sorted_train_errors)
    sorted_test_errors = np.array(sorted_test_errors)
    #
    indices = list(range(0,n))
    #
    nb_params_shallow = nb_params_shallow[indices]
    sorted_train_errors = sorted_train_errors[indices]
    sorted_test_errors = sorted_test_errors[indices]
    worst = np.max( np.concatenate( (sorted_train_errors,sorted_test_errors) ) )
    #nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    print('nb_params_shallow = ', nb_params_shallow)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    krls.plot_values(nb_params_shallow,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Shallow Neural Net (NN)',markersize=3,colour='b')
    #krls.plot_values(nb_params_shallow,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test error for Shallow NN',markersize=3,colour='b')

    path_to_experiments_BT = '../../simulation_results_scripts/%s/task_Dec_6_BT_256D_Adam_xavier_relu_N60000'%experiment_name
    path_to_experiments_BT = '../../simulation_results_scripts/%s/task_Jan_19_BT_256D_Adam_xavier_relu_N60000'%experiment_name
    path_to_experiments_BT = '../../simulation_results_scripts/%s/task_Jan_19_BT_256D_Adam_xavier_relu_N60000_original_setup'%experiment_name
    #mtf.combine_errors_and_hps_to_one_json_file(path_to_experiments_BT,verbose=True,overwrite_old=True)
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_BT,decider,verbose=False,mdl_complexity_criteria='nb_params',json_string='json')
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    nb_params_bt = [ nb_units for nb_units in sorted_units ]
    #
    nb_params_bt = np.array(nb_params_bt)
    sorted_train_errors = np.array(sorted_train_errors)
    sorted_test_errors = np.array(sorted_test_errors)
    #
    #indices = list(range(0,n))
    indices = list(range(0,n))
    #
    nb_params_bt = nb_params_bt[indices]
    sorted_train_errors = sorted_train_errors[indices]
    sorted_test_errors = sorted_test_errors[indices]
    #nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    print('nb_params_bt = ', nb_params_bt)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    #
    krls.plot_values(nb_params_bt,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Binary Tree (BT) Neural Net',markersize=3,colour='c',linestyle='--')
    #krls.plot_values(nb_params_bt,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    #
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.figure()
    plt.show()

if __name__ == '__main__':
    shallow_vs_deep()
