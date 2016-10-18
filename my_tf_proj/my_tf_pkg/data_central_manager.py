import numpy as np
import json
from sklearn.cross_validation import train_test_split

import os

from tensorflow.examples.tutorials.mnist import input_data

import pdb

def get_experiment_folder(data_file_name):
    #print( task_name.split('task_') )
    #task_prefix, task = task_name.split('task_')
    return 'om_'+data_file_name

def get_data(arg):
    '''
    Hanldes getting a data set.
    '''
    if not 'MNIST' in arg.data_file_name:
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(dirpath=arg.data_dirpath,file_name=arg.data_file_name)
    else:
        X_train, Y_train, X_cv, Y_cv, X_test, Y_tes = handle_mnist_data_set(arg)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

def get_data_from_file(dirpath,file_name):
    '''
    Gets data from file_name at dirpath. Essentially calls load(dirpath+file_name)

    dirpath = path (e.g ./data/)
    file_name = file name (e.g. f_4D_simple_ReLu_BT )

    e.g. load(dirpath+file_name) = load("./data/f_4D_simple_ReLu_BT" )
    '''
    print( os.listdir('.') )
    npzfile = np.load(dirpath+file_name+'.npz')
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

#

def get_labels(X,Y,f):
    N_train = X.shape[0]
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def get_labels_improved(X,f):
    N_train = X.shape[0]
    Y = np.zeros( (N_train,1) )
    for i in range(N_train):
        Y[i] = f(X[i])
    return Y

def generate_data(D=1, N_train=60000, N_cv=60000, N_test=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    f = f1D_task1()
    #
    low_x = low_x_var
    high_x = high_x_var
    # train

    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,D)
    Y_train = get_labels(X_train, np.zeros( (N_train,D) ) , f)
    # CV
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,D)
    Y_cv = get_labels(X_cv, np.zeros( (N_cv,D) ), f)
    # test
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,D)
    Y_test = get_labels(X_test, np.zeros( (N_test,D) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def generate_data(N_train_var=60000, N_cv_var=60000, N_test_var=60000, low_x_var=-2*np.pi, high_x_var=2*np.pi):
    f = f1D_task1()
    #
    low_x = low_x_var
    high_x = high_x_var
    # train
    N_train = N_train_var
    X_train = low_x + (high_x - low_x) * np.random.rand(N_train,1)
    Y_train = get_labels(X_train, np.zeros( (N_train,1) ) , f)
    # CV
    N_cv = N_cv_var
    X_cv = low_x + (high_x - low_x) * np.random.rand(N_cv,1)
    Y_cv = get_labels(X_cv, np.zeros( (N_cv,1) ), f)
    # test
    N_test = N_test_var
    X_test = low_x + (high_x - low_x) * np.random.rand(N_test,1)
    Y_test = get_labels(X_test, np.zeros( (N_test,1) ), f)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def generate_data_from_krls():
    N = 60000
    low_x =-2*np.pi
    high_x=2*np.pi
    X_train = low_x + (high_x - low_x) * np.random.rand(N,1)
    X_cv = low_x + (high_x - low_x) * np.random.rand(N,1)
    X_test = low_x + (high_x - low_x) * np.random.rand(N,1)
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    Y_train = f(X_train)
    Y_cv = f(X_cv)
    Y_test = f(X_test)
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

#

def handle_mnist_data(arg):
    # TODO
    if task_name == 'task_MNIST_flat':
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        X_train, Y_train = mnist.train.images, mnist.train.labels
        X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
        X_test, Y_test = mnist.test.images, mnist.test.labels
    elif task_name == 'task_MNIST_flat_auto_encoder':
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        X_train, Y_train = mnist.train.images.astype('float64'), np.copy(mnist.train.images.astype('float64'))
        X_cv, Y_cv = mnist.validation.images.astype('float64'), np.copy(mnist.validation.images.astype('float64'))
        X_test, Y_test = mnist.test.images.astype('float64'), np.copy(mnist.test.images.astype('float64'))

def get_data_hrushikesh_exp(task_name):
    if task_name == 'task_hrushikesh':
        with open('../hrushikesh/patient_data_X_Y.json', 'r') as f_json:
            patients_data = json.load(f_json)
        X = patients_data['1']['X']
        Y = patients_data['1']['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40)
        X_cv, X_test, Y_cv, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = ( np.array(X_train), np.array(Y_train), np.array(X_cv), np.array(Y_cv), np.array(X_test), np.array(Y_test) )
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
