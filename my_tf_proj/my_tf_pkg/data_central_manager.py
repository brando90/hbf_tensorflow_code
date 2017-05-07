import numpy as np
import json
#from sklearn.cross_validation import train_test_split

import re

import os

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)
import unittest

import pdb

def get_experiment_folder(data_filename):
    #print( task_name.split('task_') )
    #task_prefix, task = task_name.split('task_')
    return 'om_'+data_filename

def classification_task_or_not(arg):
    if arg.data_filename == 'MNIST':
        return True
    elif arg.data_filename == 'CIFAR':
        return True
    elif bool(re.search('f_\d+D_\w+', arg.data_filename)):
        return False
    else:
        raise ValueError('Need to use valid data set')

def get_data(arg,N_frac=60000):
    '''
    Hanldes getting a data set. Also gets the number of data points specified.

    If not stated gets the first 60000 data points.
    '''
    if 'MNIST' in arg.data_filename:
        mnist = input_data.read_data_sets(arg.data_dirpath+arg.data_filename,one_hot=True)
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mnist.train.images,mnist.train.labels, mnist.validation.images,mnist.validation.labels, mnist.test.images,mnist.test.labels
    elif 'CIFAR' in arg.data_filename:
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = handle_cifar(arg) #TODO
    else:
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(dirpath=arg.data_dirpath,filename=arg.data_filename)
    # get the number/fraction of the data set that we are going to actully use
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = X_train[:N_frac,:], Y_train[:N_frac,:], X_cv[:N_frac,:], Y_cv[:N_frac,:], X_test[:N_frac,:], Y_test[:N_frac,:]
    # set the data lengths
    arg.N_train, arg.D = X_train.shape
    arg.N_cv = X_cv.shape[0]
    arg.N_test, arg.D_out = Y_test.shape
    ##
    arg.X_train, arg.Y_train, arg.X_cv, arg.Y_cv, arg.X_test, arg.Y_test = X_train, Y_train, X_cv, Y_cv, X_test, Y_test
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

def get_data_from_file(dirpath,filename):
    '''
    Gets data from filename at dirpath. Essentially calls load(dirpath+filename)

    dirpath = path (e.g ./data/)
    filename = file name (e.g. f_4D_simple_ReLu_BT )

    e.g. load(dirpath+filename) = load("./data/f_4D_simple_ReLu_BT" )
    '''
    print( '===> os.listdir(.): ', os.listdir('.') )
    print( 'os.getcwd(): ', os.getcwd())
    print('os.path.dirname(os.path.abspath(__file__)): ', os.path.dirname(os.path.abspath(__file__)))
    #pdb.set_trace()
    npzfile = np.load(dirpath+filename+'.npz')
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)

def get_cifar():
    # TODO
    return None

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

#

# def radamacher_to_one_hot(y):
#     print(y)
#     N,D = y.shape
#     out = np.zeros([N,2],dtype=int)
#     hot_one_indices = (y==1).astype(int)
#     out[np.arange(N), hot_one_indices ] = 1
#     print(hot_one_indices)
#     return out
#
# def initialization_based(y):
#     out = np.zeros((len(y),2),dtype=int)
#     out[np.arange(out.shape[0]), (y==1).astype(int)] = 1
#     return out

def radamacher_to_one_hot(y):
    out = np.zeros((len(y),2),dtype=int)
    mask = (y == -1)
    mask = np.reshape( mask, [len(y),] )
    out[:,0] = mask.astype(int)
    out[:,1] = (~mask).astype(int)
    return out

class Test_Data_center_manager(unittest.TestCase):
    #make sure methods start with word test

    def test_to_one_hot(self):
        X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        Y = np.reshape(np.prod(X,axis=1),[4,1])
        y_ans = np.array([ [0,1], [1,0], [1,0], [0,1] ])
        Y_one_hot = radamacher_to_one_hot(Y)
        self.assertTrue( np.array_equal(Y_one_hot,y_ans) )

##

if __name__ == '__main__':
    unittest.main()
