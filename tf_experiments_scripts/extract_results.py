import json
import os
import pdb

import numpy as np
import re

import my_tf_pkg as mtf

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
    print('nn1_list_train_errors: ', nn1_list_train_errors)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    #print 'nn2_list_train_errors: ', nn2_list_train_errors
    #print 'nn2_list_test_errors: ', nn2_list_test_errors
    #
    list_units = np.array(nn1_list_units)
    print( list_units)

def display_results():
    # frameworkpython multiple_vs_single_collect_results.py
    #experiment_name = 'om_f_4d_conv'
    experiment_name = 'om_f_4d_task_conv_2nd'

    path_to_experiments = '../../%s/task_August_18_BT/August_26_jBT_12_1000_RMSProp'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_')

    #
    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    #nn2_list_units, nn2_list_train_errors, nn2_list_test_errors = get_list_errors2(experiment_results=nn2_multiple_experiment_results)
    #
    print('nn1_list_train_errors: ', nn1_list_train_errors)
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    #
    list_units = np.array(nn1_list_units)
    print( list_units)

def display_results2():
    # frameworkpython multiple_vs_single_collect_results.py
    #experiment_name = 'om_f_4d_conv'
    experiment_name = 'om_f_4d_task_conv_2nd'

    path_to_experiments = '../../%s/task_August_18_NN'%experiment_name
    nn1_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')

    path_to_experiments = '../../%s/task_August_18_BT_BN_trainable'%experiment_name
    bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    # path_to_experiments = '../../%s/task_August_18_BT'%experiment_name
    # bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')
    # path_to_experiments = '../../%s/task_August_18_BT'%experiment_name
    # bt_multiple_experiment_results = mtf.get_results_for_experiments(path_to_experiments,verbose=True, split_string='_jBT_[\d]*_|_jHBF[\d]*_|_jrun_HBF[\d]*_|jNN_')

    #
    #print nn1_multiple_experiment_results
    nn1_list_units, nn1_list_train_errors, nn1_list_test_errors = mtf.get_list_errors2(experiment_results=nn1_multiple_experiment_results)
    nn2_list_units, nn2_list_train_errors, bt_list_test_errors = mtf.get_list_errors2(experiment_results=bt_multiple_experiment_results)
    #
    print('nn1_list_test_errors: ', nn1_list_test_errors)
    print('bt_multiple_experiment_results: ', bt_list_test_errors)
    #
    # list_units = np.array(nn1_list_units)
    # print( list_units)


if __name__ == '__main__':
    #display_results_NN_xsinglog1_x()
    #display_results_NN_xsinglog1_x()
    #display_results_BT()
    display_results2()
