import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import re

import krls
import my_tf_pkg as mtf

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

if __name__ == '__main__':
    #display_results_NN_xsinglog1_x()
    #display_results_NN_xsinglog1_x()
    display_results_BT()
