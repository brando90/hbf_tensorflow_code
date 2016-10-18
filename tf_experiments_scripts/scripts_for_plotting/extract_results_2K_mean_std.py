import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls
import my_tf_pkg as mtf
import namespaces as ns

#

def display_oct_softplus_mean_std():
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

    plt.subplot(111)
    path_to_experiments = '../../%s/task_Oct_10_NN_MGD_xavier_softplus_N2000'%experiment_name
    all_results = mtf.get_all_simulation_results(path_to_experiments,decider,verbose=False)
    sorted_units, sorted_decider_mean, sorted_decider_std, sorted_test_mean, sorted_test_std = mtf.get_mean_std(all_results, decider)
    nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    print('nb_units_NN = ', sorted_units)
    print('nb_params_shallow = ', nb_params_shallow)
    print('sorted_decider_mean = ', sorted_decider_mean)
    print('sorted_decider_std = ', sorted_decider_std)
    krls.plot_errors_and_bars(nb_params_shallow,sorted_decider_mean,sorted_decider_std,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Shallow NN',markersize=3,colour='b')
    krls.plot_errors_and_bars(nb_params_shallow,sorted_test_mean,sorted_test_std,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test error for Shallow NN',markersize=3,colour='b')

    # path_to_experiments = '../../%s/task_Oct_10_BT4D_MGD_xavier_softplus_N2000'%experiment_name
    # expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments,decider,verbose=False)
    # sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    # nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    # print('nb_units_bt = ', sorted_units)
    # print('nb_params_bt = ', nb_params_bt)
    # print('sorted_train_errors = ', sorted_train_errors)
    # print('sorted_test_errors = ', sorted_test_errors)
    # krls.plot_values(nb_params_bt,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    # krls.plot_values(nb_params_bt,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    #
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.show()

if __name__ == '__main__':
    #display_test()
    #debug_plot()
    #display_results4()
    #display_results_task_f_4D_conv_2nd()
    #display_results_task_f_4D_simple_ReLu_BT()
    #display_oct_relu()
    #display_oct_softplus()
    #display_oct_softplus_mean_std()
    #display_oct_relu_noise()
    display_oct_relu()