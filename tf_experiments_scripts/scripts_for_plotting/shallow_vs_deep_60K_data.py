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
    #task_name = 'f_4D_conv_2nd'
    #task_name = 'f_4D_cos_x2_BT'
    #task_name = 'f_4D_simple_ReLu_BT_2_units_1st'
    #task_name = 'f_4D_simple_ReLu_BT'
    #task_name = 'f_4D_conv_2nd_noise_3_0_25std'
    #task_name = 'f_8D_single_relu'
    #task_name = 'f_8D_conv_cos_poly1_poly1'
    task_name = 'f_8D_conv_quad_cubic_sqrt'
    #task_name = 'f_8D_conv_quad_cubic_sqrt_shuffled'
    if '4D' in task_name:
        get_k = lambda a: 7*a + 4*3*2*a**2
        shallow = lambda k: 4*k+k+k
        bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
        get_f = lambda a: 3*2*a
    elif '8D' in task_name:
        get_k = lambda a: 13*a + 200*a*a
        shallow = lambda k: 10*k
        get_f = lambda a: 10*a
        bt = lambda F: 13*F+20*F*F
    else:
        raise ValueError('task %s not handled yet.',task_name)
    #
    experiment_name = mtf.get_experiment_folder(task_name)
    print(experiment_name)

    transform = ''
    #transform = 'log'
    plt.subplot(111)
    axes = plt.gca()
    #axes.set_ylim(bottom=-0.00002,top=0.00053)
    #path_to_experiments_NN = '../../%s/task_Oct_15_NN_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Oct_16_NN_Adam_xavier_elu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_September_9_NN_runs1000'%experiment_name
    path_to_experiments_NN = '../../%s/task_September_23_NN_xavier_softplus'%experiment_name
    #path_to_experiments_NN = '../../%s/task_September_23_NN_xavier_relu'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Oct_17_NN_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Oct_20_NN_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Oct_21_NN_Adam_xavier_softplus_N60000'%experiment_name
    path_to_experiments_NN = '../../%s/task_Oct_30_NN_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Oct_31_NN_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_NN = '../../%s/task_Nov_1_NN_Adam_xavier_relu_N60000'%experiment_name
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_NN,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    if transform == 'log':
        nb_params_shallow = [ np.log(shallow(nb_params)) for nb_params in sorted_units ]
    else:
        nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    #
    #
    nb_params_shallow = np.array(nb_params_shallow)
    sorted_train_errors = np.array(sorted_train_errors)
    sorted_test_errors = np.array(sorted_test_errors)
    #
    indices = list(range(0,n))
    #indices = [1]+list(range(4,n))
    #indices = [3]+list(range(4,n))
    #indices = list(range(4,n))
    #
    nb_params_shallow = nb_params_shallow[indices]
    sorted_train_errors = sorted_train_errors[indices]
    sorted_test_errors = sorted_test_errors[indices]
    worst = np.max( np.concatenate( (sorted_train_errors,sorted_test_errors) ) )
    nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    print('nb_params_shallow = ', nb_params_shallow)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    krls.plot_values(nb_params_shallow,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Shallow Neural Net (NN)',markersize=3,colour='b')
    #krls.plot_values(nb_params_shallow,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test error for Shallow NN',markersize=3,colour='b')

    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_Adam_BN_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_MGD_BN_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_15_BT4D_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_September_9_BTHL_runs1000'%experiment_name
    path_to_experiments_BT = '../../%s/task_September_23_BTHL_xavier_softplus'%experiment_name
    #path_to_experiments_BT = '../../%s/task_September_23_BTHL_xavier_relu'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_StMomentum_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_Nesterov_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_MGD_xavier_elu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_16_BT4D_Adam_xavier_elu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_17_BT4D_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_20_BT8D_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_21_BT8D_Adam_xavier_softplus_N60000'%experiment_name
    path_to_experiments_BT = '../../%s/task_Oct_30_BT8D_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Oct_31_BT8D_Adam_xavier_relu_N60000'%experiment_name
    #path_to_experiments_BT = '../../%s/task_Nov_1_BT8D_Adam_xavier_relu_N60000'%experiment_name
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_BT,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    #
    if transform == 'log':
        nb_params_bt = [ np.log(bt(nb_params)) for nb_params in sorted_units ]
    else:
        nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    #
    nb_params_bt = np.array(nb_params_bt)
    sorted_train_errors = np.array(sorted_train_errors)
    sorted_test_errors = np.array(sorted_test_errors)
    #
    indices = list(range(0,n))
    #indices = list(range(1,n))
    #
    nb_params_bt = nb_params_bt[indices]
    sorted_train_errors = sorted_train_errors[indices]
    sorted_test_errors = sorted_test_errors[indices]
    nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    print('nb_params_bt = ', nb_params_bt)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    #
    krls.plot_values(nb_params_bt,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    #krls.plot_values(nb_params_bt,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
    #
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend()
    plt.figure()

#######
    task_name = 'f_8D_conv_quad_cubic_sqrt_shuffled'
    experiment_name = mtf.get_experiment_folder(task_name)
    path_to_experiments_NN = '../../%s/task_Nov_1_NN_Adam_xavier_relu_N60000'%experiment_name
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_NN,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    if transform == 'log':
        nb_params_shallow = [ np.log(shallow(nb_params)) for nb_params in sorted_units ]
    else:
        nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
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
    nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    print('nb_params_shallow = ', nb_params_shallow)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    krls.plot_values(nb_params_shallow,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Shallow Neural Net (NN)',markersize=3,colour='r')
    #krls.plot_values(nb_params_shallow,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test error for Shallow NN',markersize=3,colour='b')

    path_to_experiments_BT = '../../%s/task_Nov_1_BT8D_Adam_xavier_relu_N60000'%experiment_name
    expts_best_results = mtf.get_best_results_for_experiments(path_to_experiments_BT,decider,verbose=False)
    sorted_units, sorted_train_errors, sorted_validation_errors, sorted_test_errors = mtf.get_errors_for_display(expts_best_results)
    n = len(sorted_units)
    #
    if transform == 'log':
        nb_params_bt = [ np.log(bt(nb_params)) for nb_params in sorted_units ]
    else:
        nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    #
    nb_params_bt = np.array(nb_params_bt)
    sorted_train_errors = np.array(sorted_train_errors)
    sorted_test_errors = np.array(sorted_test_errors)
    #
    indices = list(range(0,n))
    #indices = list(range(1,n))
    #
    nb_params_bt = nb_params_bt[indices]
    sorted_train_errors = sorted_train_errors[indices]
    sorted_test_errors = sorted_test_errors[indices]
    nb_params_bt = [ bt(nb_units) for nb_units in sorted_units ]
    print('nb_params_bt = ', nb_params_bt)
    print('sorted_train_errors = ', sorted_train_errors)
    print('sorted_test_errors = ', sorted_test_errors)
    #sorted_train_errors = sorted_train_errors/worst
    #sorted_test_errors = sorted_test_errors/worst
    #sorted_train_errors = np.log(sorted_train_errors)
    #sorted_test_errors = np.log(sorted_test_errors)
    #
    krls.plot_values(nb_params_bt,sorted_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Train errors for Binary Tree (BT) Neural Net',markersize=3,colour='m')
    #krls.plot_values(nb_params_bt,sorted_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Test errors for Binary Tree (BT) Neural Net',markersize=3,colour='c')
#####


    duration_type = 'hours'
    expts_best_results = mtf.get_all_simulation_results(path_to_experiments_NN,decider,verbose=False)
    sorted_units, sorted_duration_means, sorted_duration_stds = mtf.get_duration_of_experiments(all_results=expts_best_results, duration_type=duration_type)
    nb_params_shallow = [ shallow(nb_units) for nb_units in sorted_units ]
    print('sorted_duration_means ', sorted_duration_means)
    print('sorted_duration_stds ', sorted_duration_stds)
    krls.plot_errors_and_bars(nb_params_shallow,sorted_duration_means,sorted_duration_stds,xlabel='number of parameters',y_label=duration_type,label='NN Duration for experiments '+duration_type,markersize=3,colour='r')

    duration_type = 'hours'
    expts_best_results = mtf.get_all_simulation_results(path_to_experiments_BT,decider,verbose=False)
    sorted_units, sorted_duration_means, sorted_duration_stds = mtf.get_duration_of_experiments(all_results=expts_best_results, duration_type=duration_type)
    nb_params_shallow = [ bt(nb_units) for nb_units in sorted_units ]
    print('sorted_duration_means ', sorted_duration_means)
    print('sorted_duration_stds ', sorted_duration_stds)
    krls.plot_errors_and_bars(nb_params_shallow,sorted_duration_means,sorted_duration_stds,xlabel='number of parameters',y_label=duration_type,label='BT Duration for experiments '+duration_type,markersize=3,colour='m')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    shallow_vs_deep()
