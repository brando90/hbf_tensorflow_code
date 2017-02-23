import numpy as np
import json
#from sklearn.cross_validation import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

import pdb

def f1D_task1():
    # f(x) = 2*(2(cos(x)^2 - 1)^2 -1
    f = lambda x: 2*np.power( 2*np.power( np.cos(x) ,2) - 1, 2) - 1
    return f

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

def get_data_from_file(file_name):
    npzfile = np.load(file_name)
    # get data
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_cv = npzfile['X_cv']
    Y_cv = npzfile['Y_cv']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
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

##

def helloworld():
    print( 'helloworld')

def get_experiment_folder(task_name):
    print(task_name.split('task_') )
    task_prefix, task = task_name.split('task_')
    return 'om_'+ task

def get_data(task_name):
    ## Data sets
    print( '---> task_name: ', task_name)

    if task_name == 'task_qianli_func':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_1D_cos_no_noise_data.npz')
    elif task_name == 'task_f_2D_task2':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_2D_task2_ml_data_and_mesh.npz')
    elif task_name == 'task_f_2D_task2_xsinglog1_x_depth2':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_2D_task2_ml_xsinlog1_x_depth_2data_and_mesh.npz')
    elif task_name == 'task_f_2D_task2_xsinglog1_x_depth3':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_2D_task2_ml_xsinlog1_x_depth_3data_and_mesh.npz')
    elif task_name == 'task_f2D_2x2_1_cosx1x2_depth2':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_2D_2x2_1_cosx1x2_depth_2data_and_mesh.npz')
    elif task_name == 'task_f2D_2x2_1_cosx1_plus_x2_depth2':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/f_2D_2x2_1_cosx1_plus_x2_depth_2data_and_mesh.npz')
    elif task_name == 'task_h_gabor_data_and_mesh':
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = get_data_from_file(file_name='./data/h_gabor_data_and_mesh.npz')

    elif task_name == 'task_f_4D_conv':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_task_conv.npz')
    elif task_name == 'task_f_8D_conv':
        #print 'task_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8d'
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_8D_task_conv.npz')
    elif task_name == 'task_f_8D_conv_test':
        #print 'task_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8d'
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_8D_conv_test.npz')

    elif task_name == 'task_f_4D_conv_1st':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_1st.npz')

    elif task_name == 'task_f_4D_conv_2nd':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_2nd.npz')
    elif task_name == 'task_f_4D_conv_2nd_noise_3_0_25std':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_2nd_noise_3_0_25std.npz')
    elif task_name == 'task_f_4D_conv_2nd_noise_6_0_5std':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_2nd_noise_3_0_25std.npz')


    elif task_name == 'task_f_4D_conv_changing':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_changing.npz')
    elif task_name == 'task_f_4D_conv_3rd':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_3rd.npz')
    elif task_name == 'task_f_4D_conv_4th':
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_4th.npz')
    elif task_name == 'task_f_4D_conv_5th':
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_5th.npz')
    elif task_name == 'task_f_4D_conv_6th':
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_conv_6th.npz')

    elif task_name == 'task_f_4D_cos_x2_BT':
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_cos_x2_BT.npz')

    elif task_name == 'task_f_4D_non_conv':
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_task_non_conv.npz')
    elif task_name == 'task_f_8D_non_conv':
        #print 'task_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8d'
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_8D_task_non_conv.npz')
    elif task_name == 'task_f_8D_conv_cos_poly1_poly1':
        #print 'task_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8d'
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_8D_conv_cos_poly1_poly1.npz')

    elif task_name == 'task_f_4D_simple_ReLu_BT':
        #print 'task_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8dtask_f_8d'
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_data_from_file(file_name='./data/f_4D_simple_ReLu_BT.npz')

    elif task_name == 'task_MNIST_flat':
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        X_train, Y_train = mnist.train.images, mnist.train.labels
        X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
        X_test, Y_test = mnist.test.images, mnist.test.labels
    elif task_name == 'task_MNIST_flat_auto_encoder':
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        X_train, Y_train = mnist.train.images.astype('float64'), np.copy(mnist.train.images.astype('float64'))
        X_cv, Y_cv = mnist.validation.images.astype('float64'), np.copy(mnist.validation.images.astype('float64'))
        X_test, Y_test = mnist.test.images.astype('float64'), np.copy(mnist.test.images.astype('float64'))
    elif task_name == 'task_hrushikesh':
        with open('../hrushikesh/patient_data_X_Y.json', 'r') as f_json:
            patients_data = json.load(f_json)
        X = patients_data['1']['X']
        Y = patients_data['1']['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40)
        X_cv, X_test, Y_cv, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = ( np.array(X_train), np.array(Y_train), np.array(X_cv), np.array(Y_cv), np.array(X_test), np.array(Y_test) )
    else:
        raise ValueError('task_name: %s does not exist. Try experiment that exists'%(task_name))
    return (X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
