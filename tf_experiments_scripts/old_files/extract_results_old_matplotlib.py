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
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_August_9_NN1_xavier/NN2rmsprop'%experiment_name
    #path_to_experiments = '../../%s/task_August_9_NN2_Xavier_BN/NN2'%experiment_name
    nn2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
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

def display_results_multiple_vs_single():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments)
    mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = mtf.get_error_stats(multiple_experiment_results)

    experiment = '/single_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    single_experiment_results = mtf.get_results_for_experiments(path_to_experiments)
    mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = mtf.get_error_stats(single_experiment_results)
    print( mean_test_errors_single)
    print( std_test_errors_single)

    #
    list_units_multiple, list_test_errors_multiple = mtf.get_list_errors(experiment_results=multiple_experiment_results)
    list_units_single, list_test_errors_single = mtf.get_list_errors(experiment_results=single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(list_units_multiple, list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    krls.plot_errors(list_units_single, list_test_errors_single,label='HBF1 Single Errors Standard Deviations', markersize=3, colour='b')

    krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF2():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = mtf.get_error_stats(multiple_experiment_results)

    experiment = '/hbf2_multiple_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    single_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = mtf.get_error_stats(single_experiment_results)
    print( mean_test_errors_single)
    print( std_test_errors_single)

    #
    list_units_multiple, list_test_errors_multiple = mtf.get_list_errors(experiment_results=multiple_experiment_results)
    list_units_single, list_test_errors_single = mtf.get_list_errors(experiment_results=single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(list_units_multiple, list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    krls.plot_errors(2*np.array(list_units_single), list_test_errors_single,label='HBF2 Multiple Errors Standard Deviations', markersize=3, colour='b')

    #krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    #krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF1_vs_HBF1():
    ##
    experiment = '/multiple_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/single_S_dir'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf1_single_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    experiment = '/hbf2_multiple_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)

    experiment = '/hbf2_single_S'
    path_to_experiments = './om_results_test_experiments'+experiment
    hbf2_single_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_single, std_test_errors_single = get_error_stats(single_experiment_results)

    #
    hbf1_list_units_multiple, hbf1_list_test_errors_multiple = mtf.get_list_errors(experiment_results=hbf1_multiple_experiment_results)
    #hbf1_list_units_single, hbf1_list_test_errors_single = get_list_errors(experiment_results=hbf1_single_experiment_results)

    hbf2_list_units_multiple, hbf2_list_test_errors_multiple = mtf.get_list_errors(experiment_results=hbf2_multiple_experiment_results)
    #hbf2_list_units_single, hbf2_list_test_errors_single = get_list_errors(experiment_results=hbf2_single_experiment_results)
    #
    plt.figure(3)
    krls.plot_errors(hbf1_list_units_multiple, hbf1_list_test_errors_multiple,label='HBF1 Multiple Standard Deviations', markersize=3, colour='r')
    #krls.plot_errors(hbf1_list_units_multiple, hbf1_list_test_errors_single,label='HBF1 Single Errors Standard Deviations', markersize=3, colour='m')

    print( len(hbf2_list_test_errors_multiple))
    print( len(hbf1_list_units_multiple))
    krls.plot_errors(2*np.array(hbf2_list_units_multiple), hbf2_list_test_errors_multiple,label='HBF2 Multiple Standard Deviations', markersize=3, colour='b')
    #krls.plot_errors(2*np.array(hbf2_list_units_multiple), hbf2_list_test_errors_single,label='HBF2 Single Errors Standard Deviations', markersize=3, colour='c')

    #krls.plot_errors_and_bars(list_units_multiple, mean_test_errors_multiple, std_test_errors_multiple, label='Multiple Errors', markersize=3, colour='b')
    #krls.plot_errors_and_bars(list_units_single, mean_test_errors_single, std_test_errors_single, label='Single Errors', markersize=3, colour='r')
    #
    plt.legend()
    plt.show()

def display_results_HBF1_task2():
    ##
    # experiment = '/multiple_S_task2_HP_hbf2'
    # path_to_experiments = './om_results_test_experiments'+experiment
    # hbf1_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True)
    #mean_train_errors, std_train_errors, mean_test_errors_multiple, std_test_errors_multiple = get_error_stats(multiple_experiment_results)

    path_to_experiments = './om_results_test_experiments/multiple_S_task2_HP_hbf2'
    hbf1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)
    #print hbf1_multiple_experiment_results

    #
    list_units, list_train_errors, list_test_errors = mtf.get_list_errors2(experiment_results=hbf1_multiple_experiment_results)
    #hbf1_list_units_single, hbf1_list_test_errors_single = get_list_errors(experiment_results=hbf1_single_experiment_results)

    #
    plt.figure(3)
    print( 'hbf1_list_units_multiple: ', list_units)
    print( 'list_train_errors: ', list_train_errors)
    print( 'list_test_errors: ', list_test_errors)
    krls.plot_errors(list_units, list_train_errors,label='HBF1 not shared HBF shape', markersize=3, colour='b')
    krls.plot_errors(list_units, list_test_errors,label='HBF1 not shared HBF shape', markersize=3, colour='r')

    plt.legend()
    plt.show()

def display_results_NN_xsinglog1_x():
    # frameworkpython multiple_vs_single_collect_results.py
    #print os.path.isdir(path_to_experiments)
    experiment_name = 'om_xsinlog1_x_depth2'
    path_to_experiments = '../../%s/task_27_july_NN1_depth_2_1000'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)

    path_to_experiments = '../../%s/task_27_july_NN2_depth_2_1000'%experiment_name
    #path_to_experiments = '../../%s/task_28_july_NN2_1000_BN'%experiment_name
    #path_to_experiments = '../../%s/task_28_july_NN2_1000_BN_False_trainable_BN'%experiment_name
    nn2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)

    path_to_experiments = '../../%s/task_27_july_NN3_depth_2_1000'%experiment_name
    #path_to_experiments = '../../%s/task_28_july_NN3_1000_BN'%experiment_name
    #path_to_experiments = '../../%s/task_28_july_NN3_1000_BN_False_trainable_BN'%experiment_name
    nn3_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True)

    #
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = mtf.get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    nn3_list_units, nn3_list_train_errors, nn3_list_test_errors = mtf.get_list_errors2(experiment_results=nn3_multiple_experiment_results)

    #
    plt.figure(3)
    #
    list_units = np.array(nn1_list_units)
    print( list_units)
    #krls.plot_errors(list_units, nn1_list_train_errors,label='NN1 train', markersize=3, colour='b')
    krls.plot_errors(list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='c')
    #
    list_units = 2*np.array(nn2_list_units[0:1]+nn2_list_units[2:])
    print( list_units)
    #krls.plot_errors(list_units, nn2_list_train_errors,label='NN2 train', markersize=3, colour='r')
    krls.plot_errors(list_units, nn2_list_test_errors[0:1]+nn2_list_test_errors[2:],label='NN2 test', markersize=3, colour='m')
    #
    list_units = 3*np.array(nn3_list_units)
    print( list_units)
    #krls.plot_errors(list_units, nn3_list_train_errors,label='NN3 train', markersize=3, colour='g')
    #krls.plot_errors(list_units, nn3_list_test_errors,label='NN3 test', markersize=3, colour='y')

    plt.legend()
    plt.show()

def display_results_hbf_xsinglog1_x():
    # frameworkpython multiple_vs_single_collect_results.py
    #print os.path.isdir(path_to_experiments)
    #experiment_name = 'om_xsinlog1_x_depth2_hbf'
    #experiment_name = 'om_2x2_1_cosx1_plus_x2_depth2'
    experiment_name = 'om_mnist'
    #path_to_experiments = '../../%s/task_30_july_HBF1_depth_2_1000'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000_Xavier'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000_BN_true_true'%experiment_name
    #path_to_experiments = '../../%s/task_August_HBF1_depth_2_1000_dont_train_S/hbf1_dont_train'%experiment_name
    path_to_experiments = '../../%s/task_August_HBF1_depth_2_1000_dont_train_S_gpu/August_05_jrun_HBF1_12_multiple_1000'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #path_to_experiments = '../../%s/task_30_july_HBF2_depth_2_1000'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000_Xavier'%experiment_name
    path_to_experiments = '../../%s/task_1_August_HBF2_depth_2_1000_BN_true_true'%experiment_name
    path_to_experiments = '../../%s/task_mnist_August_7_HBF2_dont_train_S_data_trunc_norm_kern_30'%experiment_name
    nn2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #path_to_experiments = '../../%s/task_30_july_HBF2_depth_2_1000'%experiment_name
    #nn3_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jNN[\d]*_')

    #
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = mtf.get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    #nn3_list_units, nn3_list_train_errors, nn3_list_test_errors = get_list_errors2(experiment_results=nn3_multiple_experiment_results)

    #
    plt.figure(3)
    #
    nn1_list_train_errors, nn1_list_test_errors = 784*np.array(nn1_list_train_errors), 784*np.array(nn1_list_test_errors)
    nn2_list_train_errors, nn2_list_test_errors = np.array(nn2_list_train_errors), np.array(nn2_list_test_errors)
    print( 'nn1_list_train_errors: ', nn1_list_train_errors)
    print( 'nn1_list_test_errors: ', nn1_list_test_errors)
    print( 'nn2_list_train_errors: ', nn2_list_train_errors)
    print( 'nn2_list_test_errors: ', nn2_list_test_errors)
    list_units = np.array(nn1_list_units)
    print( list_units)
    krls.plot_errors(list_units, nn1_list_train_errors,label='HBF1 train', markersize=3, colour='b')
    krls.plot_errors(list_units, nn1_list_test_errors,label='HBF1 test', markersize=3, colour='c')
    #
    list_units = 2*np.array(nn2_list_units)
    print( list_units)
    krls.plot_errors(list_units, nn2_list_train_errors,label='HBF2 train', markersize=3, colour='r')
    krls.plot_errors(list_units, nn2_list_test_errors,label='HBF2 test', markersize=3, colour='m')
    #
    #list_units = 3*np.array(nn3_list_units)
    #print list_units
    #krls.plot_errors(list_units, nn3_list_train_errors,label='NN3 train', markersize=3, colour='g')
    #krls.plot_errors(list_units, nn3_list_test_errors,label='NN3 test', markersize=3, colour='y')

    plt.legend()
    plt.show()

def display_results_BT():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4d_conv'

    path_to_experiments = '../../%s/task_August_13_BT/BT_6_12_18'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #path_to_experiments = '../../%s/task_August_13_BT/August_25_jBT_12_1000_RMSProp'%experiment_name
    #nn2_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #
    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    #nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    #
    plt.figure(3)
    print('nn1_list_train_errors: ', nn1_list_train_errors)
    print( 'nn1_list_test_errors: ', nn1_list_test_errors)
    #print 'nn2_list_train_errors: ', nn2_list_train_errors
    #print 'nn2_list_test_errors: ', nn2_list_test_errors
    #
    list_units = np.array(nn1_list_units)
    print( list_units)
    krls.plot_errors(list_units, nn1_list_train_errors,label='BT train', markersize=3, colour='b')
    krls.plot_errors(list_units, nn1_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    # list_units = 2*np.array(nn2_list_units)
    # print list_units
    # krls.plot_errors(list_units, nn2_list_train_errors,label='BT12 train', markersize=3, colour='r')
    # krls.plot_errors(list_units, nn2_list_test_errors,label='BT12 test', markersize=3, colour='m')
    #
    plt.legend()
    plt.show()

def display_results_BT():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4d_task_conv_2nd'

    path_to_experiments = '../../%s/task_August_13_BT/BT_6_12_18'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #path_to_experiments = '../../%s/task_August_13_BT/August_25_jBT_12_1000_RMSProp'%experiment_name
    #nn2_multiple_experiment_results = get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #
    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    #nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    #
    plt.figure(3)
    print('nn1_list_train_errors: ', nn1_list_train_errors)
    print( 'nn1_list_test_errors: ', nn1_list_test_errors)
    #print 'nn2_list_train_errors: ', nn2_list_train_errors
    #print 'nn2_list_test_errors: ', nn2_list_test_errors
    #
    list_units = np.array(nn1_list_units)
    print( list_units)
    krls.plot_errors(list_units, nn1_list_train_errors,label='BT train', markersize=3, colour='b')
    krls.plot_errors(list_units, nn1_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    # list_units = 2*np.array(nn2_list_units)
    # print list_units
    # krls.plot_errors(list_units, nn2_list_train_errors,label='BT12 train', markersize=3, colour='r')
    # krls.plot_errors(list_units, nn2_list_test_errors,label='BT12 test', markersize=3, colour='m')
    #
    plt.legend()
    plt.show()

def display_results2():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4d_conv_changing'
    #experiment_name = 'om_f_4d_task_conv_2nd'


    path_to_experiments = '../../%s/task_August_29_NN'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_August_30_BT'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print()
    print('nn1_list_test_errors = ', nn1_list_test_errors)
    print('bt_multiple_experiment_results = ', bt_list_test_errors)
    #
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 train', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results3():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4D_conv_6th'
    #experiment_name = 'om_f_4d_task_conv_2nd'


    path_to_experiments = '../../%s/task_August_34_NN'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_August_34_BT'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 train', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results5():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_mnist'
    #experiment_name = 'om_f_4d_task_conv_2nd'


    path_to_experiments = '../../%s/task_August_7_NN1_xavier_momentum'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_August_9_NN1_xavier/NN2rmsprop'%experiment_name
    path_to_experiments = '../../%s/task_August_9_NN2_Xavier_BN/NN2'%experiment_name
    nn2_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(nn2_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = mtf.get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    #
    print('units: ', nn1_list_units)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', nn2_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    # krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='b')
    # krls.plot_errors(nn1_list_units, nn2_list_test_errors,label='NN2 test', markersize=3, colour='c')
    krls.plot_values(nn1_list_units,nn1_list_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN1 test',markersize=3,colour='b')
    krls.plot_values(nn1_list_units,nn2_list_test_errors,xlabel='number of units',y_label='squared error (l2 loss)',label='NN2 test',markersize=3,colour='c')
    #
    plt.legend()
    plt.show()

def display_results6():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4d_conv_2nd'

    path_to_experiments = '../../%s/task_September_8_NN_100'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_8_BTHL_100'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('units: ', nn1_list_units)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results7():
    # frameworkpython multiple_vs_single_collect_results.py
    experiment_name = 'om_f_4d_conv_2nd'

    path_to_experiments = '../../%s/task_September_9_BTHL_runs1000'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_9_NN_runs1000'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('units: ', nn1_list_units)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results8():
    task_name = 'task_f_4d_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)
    #experiment_name = 'om_f_4d_conv_2nd'
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_9_BTHL_runs1000'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_9_NN_runs1000'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('units: ', nn1_list_units)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results9():
    task_name = 'task_f_4d_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)
    #experiment_name = 'om_f_4d_conv_2nd'
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_14_NN_run100_elu_initXav'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_14_BTHL_runs100_elu_initXav'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('units: ', nn1_list_units)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)
    krls.plot_errors(nn1_list_units, nn1_list_test_errors,label='NN1 test', markersize=3, colour='b')
    krls.plot_errors(nn1_list_units, bt_list_test_errors,label='BT test', markersize=3, colour='c')
    #
    plt.legend()
    plt.show()

def display_results10():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a

    task_name = 'task_f_4d_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)
    #experiment_name = 'om_f_4d_conv_2nd'
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_14_NN_runs200_xavier_softplus'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_14_BTHL_runs200_xavier_softplus'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(BT)', len(bt_multiple_experiment_results))

    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    bt_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    nb_params = [ shallow(nb_units) for nb_units in nn1_list_units ]
    print('shallow units = ', nn1_list_units)
    print('bt_list_units = ', bt_list_units)
    print('nn1_list_test_errors = ', nn1_list_test_errors)
    print('bt_multiple_experiment_results = ', bt_list_test_errors)
    print('nb_params = ', nb_params)
    krls.plot_values(nb_params,nn1_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN test',markersize=3,colour='b')
    krls.plot_values(nb_params,bt_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN test',markersize=3,colour='c')
    #
    plt.legend()
    plt.show()

def display_results11():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a
    #
    get_errors_from = mtf.get_errors_based_on_train_error

    task_name = 'task_f_4D_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)

    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_14_NN_runs200_xavier_softplus'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_14_BTHL_runs200_xavier_softplus'%experiment_name
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
    krls.plot_values(nb_params_shallow,nn1_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN train',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN train',markersize=3,colour='c')
    # plot test errors
    krls.plot_values(nb_params_shallow,nn1_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN test',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN test',markersize=3,colour='c')
    #
    plt.legend()
    plt.show()

def display_results_task_f_4D_conv_2nd():
    get_k = lambda a: 7*a + 4*3*2*a**2
    shallow = lambda k: 4*k+k+k
    bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)
    get_f = lambda a: 3*2*a
    #
    get_errors_from = mtf.get_errors_based_on_train_error
    get_errors_from = mtf.get_errors_based_on_validation_error

    task_name = 'task_f_4D_conv_2nd'
    experiment_name = mtf.get_experiment_folder(task_name)

    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    var_Y_test = np.var(Y_test)
    print('var_Y_test: ', var_Y_test)

    path_to_experiments = '../../%s/task_September_23_NN_xavier_softplus'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,get_errors_from,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    print('LEN(NN)', len(nn1_multiple_experiment_results))

    path_to_experiments = '../../%s/task_September_23_BTHL_xavier_softplus'%experiment_name
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
    krls.plot_values(nb_params_shallow,nn1_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN train',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_train_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN train',markersize=3,colour='c')
    # plot test errors
    krls.plot_values(nb_params_shallow,nn1_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Shallow NN test',markersize=3,colour='b')
    krls.plot_values(nb_params_bt,bt_list_test_errors,xlabel='number of parameters',y_label='squared error (l2 loss)',label='Binary Tree NN test',markersize=3,colour='c')
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
    decider = ns.Namespace(get_errors_from=get_errors_from)
    #
    experiment_name = 'om_mnist'

    path_to_experiments = '../../%s/task_August_7_NN1_xavier_momentum'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,decider,verbose=False)

    path_to_experiments = '../../%s/task_August_9_NN1_xavier/NN2rmsprop'%experiment_name


if __name__ == '__main__':
    display_test()
    #debug_plot()
    #display_results4()
    #display_results_task_f_4D_conv_2nd()
    #display_results_task_f_4D_simple_ReLu_BT()