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

def debug_plot():
    # plt.subplot(211)
    # plt.plot([1,2,3], label="test1")
    # plt.plot([3,2,1], label="test2")
    # # Place a legend above this subplot, expanding itself to
    # # fully use the given bounding box.
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(121)
    plt.plot([1,2,3], label="test1")
    plt.plot([3,2,1], label="test2")
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

def display_results4():
    get_errors_from = mtf.get_errors_based_on_train_error
    #
    experiment_name = 'om_mnist'

    path_to_experiments = '../../%s/task_August_7_NN1_xavier_momentum'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True)
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_August_9_NN1_xavier/NN2rmsprop'%experiment_name
    #path_to_experiments = '../../%s/task_August_9_NN2_Xavier_BN/NN2'%experiment_name
    nn2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True)
    print('LEN(BT)', len(nn2_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors(experiment_results=nn1_multiple_experiment_results,get_errors_from=get_errors_from)
    nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = mtf.get_list_errors(experiment_results=nn2_multiple_experiment_results,get_errors_from=get_errors_from)
    #
    print('get_errors_from = ', get_errors_from.__name__)
    print('units = ', nn1_list_units)
    print('nn1_list_test_errors = ', nn1_list_test_errors)
    print('bt_multiple_experiment_results = ', nn2_list_test_errors)
    #
    plt.subplot(121)
    krls.plot_values(nn1_list_units,nn1_list_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN1 test',markersize=3,colour='b')
    krls.plot_values(nn1_list_units,nn2_list_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN2 test',markersize=3,colour='c')
    #
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.show()

##

def display_results_task_f_4D_conv_2nd():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a
    #
    task_name = 'task_f_4D_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)

    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors(experiment_results=nn1_multiple_experiment_results,get_errors_from=get_errors_from)
    bt_list_units, bt_list_train_errors, bt_list_test_errors = mtf.get_list_errors(experiment_results=bt_multiple_experiment_results,get_errors_from=get_errors_from)
    #

    get_errors_from = mtf.get_errors_based_on_train_error
    #get_errors_from = mtf.get_errors_based_on_validation_error
    decider = ns.Namespace(get_errors_from=get_errors_from)

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

def display_results_task_f_4D_simple_ReLu_BT():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a
    #
    get_errors_from = mtf.get_errors_based_on_train_error
    get_errors_from = mtf.get_errors_based_on_validation_error

    task_name = 'task_f_4D_simple_ReLu_BT'
    experiment_name = mtf.get_experiment_folder(task_name)

    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_23_NN_xavier_relu'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_23_BTHL_xavier_relu'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors(experiment_results=nn1_multiple_experiment_results,get_errors_from=get_errors_from)
    bt_list_units, bt_list_train_errors, bt_list_test_errors = mtf.get_list_errors(experiment_results=bt_multiple_experiment_results,get_errors_from=get_errors_from)
    #
    nb_params_shallow = [ shallow(nb_units) for nb_units in nn1_list_units ]
    nb_params_bt = [ bt(nb_units) for nb_units in bt_list_units ]
    print('get_errors_from = ', get_errors_from.__name__)

    print('shallow units = ', nn1_list_units)
    print('bt_list_units = ', bt_list_units)

    print('nb_params_shallow = ', nb_params_shallow)
    print('nb_params_bt = ', nb_params_bt)

    print('nn1_list_train_errors = ', nn1_list_train_errors)
    print('bt_list_train_errors = ', bt_list_train_errors)
    print('nn1_list_test_errors = ', nn1_list_test_errors)
    print('bt_multiple_experiment_results = ', bt_list_test_errors)
    # plot train errors
    plt.subplot(121)
    krls.plot_values(nb_params_shallow,nn1_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN train',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN train',markersize=3,colour='c')
    # plot test errors
    krls.plot_values(nb_params_shallow,nn1_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN test',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN test',markersize=3,colour='c')
    #
    plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def display_test():
    get_errors_from = mtf.get_errors_based_on_train_error
    #get_errors_from = mtf.get_errors_based_on_validation_error
    decider = ns.Namespace(get_errors_from=get_errors_from)
    #
    experiment_name = 'om_mnist'

    path_to_experiments = '../../%s/task_August_7_NN1_xavier_momentum'%experiment_name
    expts_best_results1 = mtf.get_results_for_experiments(path_to_experiments,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results1)
    krls.plot_values(sorted_units,sorted_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN1 test',markersize=3,colour='b')

    path_to_experiments = '../../%s/task_August_9_NN1_xavier/NN2rmsprop'%experiment_name
    expts_best_results1 = mtf.get_results_for_experiments(path_to_experiments,decider,verbose=False)
    _, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results1)
    krls.plot_values(sorted_units,sorted_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN2 test',markersize=3,colour='c')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    display_test()
    #debug_plot()
    #display_results4()
    #display_results_task_f_4D_conv_2nd()
    #display_results_task_f_4D_simple_ReLu_BT()
