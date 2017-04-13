import tensorflow as tf

#from sklearn import preprocessing
import numpy as np

import random

import shutil
import subprocess
import json
import sys
import datetime
import os
import pdb
import ast
import pickle
import csv
import copy
import functools

import my_tf_pkg as mtf
import sgd_lib
import time

import maps
import functools

from tensorflow.python.client import device_lib

print_func_flush_true = functools.partial(print, flush=True) # TODO fix hack
print_func_flush_false = functools.partial(print, flush=False) # TODO fix hack

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

##

def deleteContent(pfile):
    pfile.seek(0)
    pfile.truncate()

def is_jsonable(x):
    '''
    checks if x is json dumpable
    '''
    try:
        json.dumps(x) # Serialize obj to a JSON formatted str using this conversion table.
        return True
    except:
        return False

def get_remove_functions_from_dict(arg_dict):
    '''
        Removes functions from dictionary and returns modified dictionary
    '''
    #arg_dict = copy.deepcopy(arg_dict)
    arg_dict_copy = copy.copy(dict(arg_dict))
    keys_to_delete = []
    for key,value in arg_dict_copy.items():
        if hasattr(value, '__call__') or not is_jsonable(value):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del arg_dict_copy[key]
    return arg_dict_copy

#

def preprocess_data(arg, X_train, Y_train, X_cv, Y_cv, X_test, Y_test):
    if arg.type_preprocess_data =='re_shape_X_to_(N,1,D,1)':
        X_train, X_cv, X_test = X_train.reshape(arg.N_train,1,arg.D,1), X_cv.reshape(arg.N_cv,1,arg.D,1), X_test.reshape(arg.N_test,1,arg.D,1)
    elif arg.type_preprocess_data == 'flat_autoencoder':
        # TODO fix and see if its still neccessary
        if arg.data_file_name == 'task_MNIST_flat_auto_encoder':
            with tf.name_scope('input_reshape'):
                x_image = tf.to_float(x, name='ToFloat')
                image_shaped_input_x = tf.reshape(x_image, [-1, 28, 28, 1])
                # tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)
                tf.image_summary('input', image_shaped_input_x, 10)

            with tf.name_scope('reconstruct'):
                y_image = tf.to_float(y, name='ToFloat')
                image_shaped_input_y = tf.reshape(x_image, [-1, 28, 28, 1])
                # tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)
                tf.image_summary('reconstruct', image_shaped_input_y, 10)
    else:
        raise ValueError('This type of preprocessing has not bee implemented ',+arg.type_preprocess_data)
    return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
#

def count_number_trainable_params(y):
    '''
    Receives model y=mdl tf graph/thing/object and counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        #print('trainable_variable ', trainable_variable.__dict__)
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

def get_mdl(arg,x):
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = arg.get_dataset(arg)
    ## Make Model
    if arg.mdl == 'debug_mdl':
        W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
        b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    elif arg.mdl == 'standard_nn':
        #
        arg.mu_init_list = arg.get_W_mu_init(arg)
        arg.std_init_list = arg.get_W_std_init(arg)
        arg.b_init = arg.get_b_init(arg)

        arg.dims = arg.get_dims(arg)
        nb_layers = len(arg.dims)-1
        nb_hidden_layers = nb_layers-1
        inits_C,inits_W,inits_b = mtf.get_initilizations_standard_NN(init_type=arg.init_type,dims=arg.dims,mu=arg.mu_init_list,std=arg.std_init_list,b_init=arg.b_init, X_train=X_train, Y_train=Y_train)
        with tf.name_scope("standardNN") as scope:
            y = mtf.build_standard_NN(arg, x,arg.dims,(None,inits_W,inits_b))
            y = mtf.get_summation_layer(l=str(nb_layers),x=y,init=inits_C[0])
    elif arg.mdl == 'hbf':
        raise ValueError('HBF not implemented yet.')
        # arg.dims = [D]+arg.units+[D_out]
        # trainable_S = True if (arg.trainable_S=='train_S') else False
        # arg.b_init = arg.get_b_init(arg)
        # arg.S_init = arg.b_init
        # float_type = tf.float64
        # #arg.mu , arg.std = arg.get_W_mu_init(arg), arg.get_W_std_init(arg)
        # x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D
        # (inits_C,inits_W,inits_S,rbf_error) = mtf.get_initilizations_HBF(init_type=arg.init_type,dims=arg.dims,mu=arg.mu,std=arg.std,b_init=arg.b_init,S_init=arg.S_init, X_train=X_train, Y_train=Y_train, train_S_type=arg.train_S_type)
        # #print(inits_W)
        # nb_layers = len(arg.dims)-1
        # nb_hidden_layers = nb_layers-1
        # with tf.name_scope("HBF") as scope:
        #     mdl = mtf.build_HBF2(x,arg.dims,(inits_C,inits_W,inits_S),phase_train,arg.trainable_bn,trainable_S)
        #     mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
    elif arg.mdl == 'bt_subgraph':
        # note: x is shape [None,1,D,1]
        with tf.name_scope("mdl"+arg.scope_name) as scope:
            y = mtf.bt_mdl_conv_subgraph(arg,x)
    elif 'binary_tree' in arg.mdl:
        with tf.name_scope("mdl"+arg.scope_name) as scope:
            y = mtf.bt_mdl_conv(arg,x)
    # elif arg.mdl == 'basin_expt':
    #     with tf.name_scope("mdl"+arg.scope_name) as scope:
    #         y = mtf.bt_mdl_conv(arg,x)

    arg.nb_params = count_number_trainable_params(y)
    return y

##
def get_optimizer(arg):
    ### set up optimizer from args
    arg.nb_steps = arg.get_steps(arg)
    arg.batch_size = arg.get_batch_size(arg)
    arg.log_learning_rate = arg.get_log_learning_rate(arg)
    #pdb.set_trace()
    arg.starter_learning_rate = arg.get_start_learning_rate(arg)
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    arg.decay_rate = arg.get_decay_rate(arg)
    arg.decay_steps = arg.get_decay_steps(arg)
    if arg.optimization_alg == 'GD':
        pass
    elif arg.optimization_alg =='Momentum':
        arg.use_nesterov = arg.get_use_nesterov()
        arg.momentum = arg.get_momentum(arg)
        print('arg.use_nesterov', arg.use_nesterov)
        print('arg.momentum', arg.momentum)
    elif arg.optimization_alg == 'Adadelta':
        arg.rho = arg.get_rho(arg)
        print('arg.rho', arg.rho)
    elif arg.optimization_alg == 'Adagrad':
        #only has learning rate
        pass
    elif arg.optimization_alg == 'Adam':
        arg.beta1 = arg.get_beta1(arg)
        arg.beta2 = arg.get_beta2(arg)
        print('arg.beta1', arg.beta1)
        print('arg.beta2', arg.beta2)
    elif arg.optimization_alg == 'RMSProp':
        arg.decay = arg.get_decay(arg)
        arg.momentum = arg.get_momentum(arg)
        print('arg.decay', arg.decay)
        print('arg.momentum', arg.momentum)
    elif arg.optimization_alg == 'GDL':
        arg.mu_noise = arg.get_gdl_mu_noise(arg)
        arg.stddev_noise = arg.get_gdl_stddev_noise(arg)
    else:
        raise ValueError('Invalid optimizer. Make sure you are using an optimizer that exists.')
    with tf.name_scope("train") as scope:
        # If the argument staircase is True, then global_step / decay_steps is an integer division and the decayed earning rate follows a staircase function.
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        arg.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=arg.starter_learning_rate, global_step=arg.global_step,decay_steps=arg.decay_steps, decay_rate=arg.decay_rate, staircase=arg.staircase)
        arg.learning_rate = learning_rate
        # Passing global_step to minimize() will increment it at each step.
        if arg.optimization_alg == 'GD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif arg.optimization_alg == 'Momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=arg.momentum,use_nesterov=arg.use_nesterov)
        elif arg.optimization_alg == 'Adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=arg.rho, epsilon=1e-08, use_locking=False, name='Adadelta')
        elif arg.optimization_alg == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=arg.beta1, beta2=arg.beta2, epsilon=1e-08, name='Adam')
        elif arg.optimization_alg == 'Adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif arg.optimization_alg == 'RMSProp':
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=arg.decay, momentum=arg.momentum, epsilon=1e-10, name='RMSProp')
        elif arg.optimization_alg == 'GDL':
            #opt = sgd_lib.GDL(learning_rate,mu_noise=arg.mu_noise,stddev_noise=arg.stddev_noise)
            #opt = sgd_lib.GDL_official_tf(loss=arg.loss,learning_rate=learning_rate,mu_noise=arg.mu_noise,stddev_noise=arg.stddev_noise,compact=arg.compact,B=arg.B)
            opt = sgd_lib.GDL_official_tf(arg)
    train_step = opt
    return train_step

##

def main_hp_serial(arg):
    #do jobs
    SLURM_ARRAY_TASK_IDS = list(range(int(arg.nb_array_jobs)))
    for job_array_index in SLURM_ARRAY_TASK_IDS:
        scope_name = 'stid_'+str(job_array_index)
        #with tf.name_scope(scope_name):
        with tf.variable_scope(scope_name):
            #arg = arg.get_arg_for_experiment()
            arg.slurm_array_task_id = job_array_index
            main_nn(arg)
##

def get_batch_feed(X, Y, batch_size):
    mini_batch_indices = np.random.randint(batch_size,size=batch_size)
    return X[mini_batch_indices,:], Y[mini_batch_indices,:]

def get_accuracy_loss(arg,x,y,y_):
    '''
    Note: when the task is regression accuracy = loss but for classification
    loss = cross_entropy,svm_loss, surrogate_loss, etc and accuracy = 1 - {0-1 loss}.
    '''
    with tf.name_scope("loss_and_acc") as scope:
        if arg.classificaton:
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
            #
            loss, accuracy = cross_entropy, tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            l2_loss = tf.reduce_sum( tf.reduce_mean(tf.square(y_-y), 0))
            #
            loss, accuracy = l2_loss, l2_loss
    return loss, accuracy

def main_hp(arg):
    '''
    executes the current hp (hyper param) slurm array task. Usually means that
    it has to either continue training a model that wasn't finished training
    or start one from scratch.

    note:
    '''
    #arg.rand_x = int.from_bytes(os.urandom(4), sys.byteorder)
    np.random.seed(arg.rand_x)
    random.seed(arg.rand_x)
    tf.set_random_seed( arg.rand_x )
    # force to flushing to output as default
    print = print_func_flush_false
    if arg.slurm_array_task_id == '1':
        #arg.display_training = True
        print = print_func_flush_true
    if arg.flush:
        print = print_func_flush_true
    print(print)
    print('>>> arg.restore = ', arg.restore)
    arg.date = datetime.date.today().strftime("%B %d").replace (" ", "_")
    #
    current_job_mdl_folder = 'job_mdl_folder_%s/'%arg.job_name
    arg.path_to_hp = arg.get_path_root(arg)+current_job_mdl_folder
    arg.path_to_ckpt = arg.get_path_root_ckpts(arg)+current_job_mdl_folder
    arg.hp_folder_for_ckpt = 'hp_stid_%s/'%str(arg.slurm_array_task_id)
    ### get folder structure for experiment
    mtf.make_and_check_dir(path=arg.get_path_root(arg)+current_job_mdl_folder)
    mtf.make_and_check_dir(path=arg.get_path_root_ckpts(arg)+current_job_mdl_folder)
    #
    #errors_pretty = '/errors_file_%s_slurm_sj%s.txt'%(arg.date,arg.slurm_array_task_id)
    arg.json_hp_filename = 'json_hp_stid%s'%(arg.slurm_array_task_id)
    arg.csv_errors_filename = 'csv_errors_slurm_array_id%s'%(arg.slurm_array_task_id)
    ##
    # if arg.restore: TODO
    #     arg = restore_hps(arg)
    #     arg.float_type = tf.float32
    ## get data set
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = mtf.get_data(arg,arg.N_frac)
    print( '(N_train,D) = (%d,%d) \n (N_test,D_out) = (%d,%d) ' % (arg.N_train,arg.D, arg.N_test,arg.D_out) )
    ## if (preprocess_data, then preprocess) else (do nothing to the data)
    if arg.type_preprocess_data:
        X_train, Y_train, X_cv, Y_cv, X_test, Y_test = preprocess_data(arg, X_train, Y_train, X_cv, Y_cv, X_test, Y_test)
    #### build graph
    graph = tf.Graph()
    with graph.as_default():
        ### get mdl
        x = tf.placeholder(arg.float_type, arg.get_x_shape(arg), name='x-input')
        y_ = tf.placeholder(arg.float_type, arg.get_y_shape(arg))
        phase_train = tf.placeholder(tf.bool, name='phase_train') # phase_train = tf.placeholder(tf.bool, name='phase_train') if arg.bn else  None
        arg.phase_train = phase_train
        y = get_mdl(arg,x)
        ### get loss and accuracy
        loss, accuracy = get_accuracy_loss(arg,x,y,y_)
        ### get optimizer variables
        opt = get_optimizer(arg)
        train_step = opt.minimize(loss, global_step=arg.global_step)
        # step for optimizer (useful for ckpts)
        step, nb_iterations = tf.Variable(0, name='step'), tf.Variable(arg.nb_steps, name='nb_iterations')
        batch_size = tf.Variable(arg.batch_size, name='batch_size')
        # save everything that was saved in the session
        saver = tf.train.Saver()
    #### run session
    arg.save_ckpt_freq = arg.get_save_ckpt_freq(arg)
    start_time = time.time()
    with tf.Session(graph=graph) as sess:
        with open(arg.path_to_hp+arg.csv_errors_filename,mode='a') as errors_csv_f: # a option: Opens a file for appending. The file pointer is at the end of the file if the file exists. That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
            #writer = csv.Writer(errors_csv_f)
            writer = csv.DictWriter(errors_csv_f,['train_error', 'cv_error', 'test_error'])
            # if (there is a restore ckpt mdl restore it) else (create a structure to save ckpt files)
            if arg.restore:
                arg.restore = False # after the model has been restored, we continue normal until all hp's are finished
                saver.restore(sess=sess, save_path=arg.save_path_to_ckpt2restore) # e.g. saver.restore(sess=sess, save_path='./tmp/my-model')
                arg = restore_hps(arg)
                print('arg ', arg)
                #print('arg.save_path_to_ckpt2restore: ',arg.save_path_to_ckpt2restore)
                print('restored model trained up to, STEP: ', step.eval())
                print('restored model, ACCURACY:', sess.run(fetches=accuracy, feed_dict={x: X_train, y_: Y_train, phase_train: False}))
            else: # NOT Restore
                # not restored, so its a virgin run from scratch for this hp
                deleteContent(pfile=errors_csv_f) # since its a virgin run we
                writer.writeheader()
                #
                save_hps(arg) # save current hyper params
                if arg.save_checkpoints or arg.save_last_mdl:
                    mtf.make_and_check_dir(path=arg.path_to_ckpt+arg.hp_folder_for_ckpt) # creates ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N
                sess.run(tf.global_variables_initializer())
            # train
            start_iteration = step.eval() # last iteration trained is the first iteration for this model
            batch_size_eval = batch_size.eval()
            #pdb.set_trace()
            for i in range(start_iteration,nb_iterations.eval()):
                #batch_xs, batch_ys = mnist.train.next_batch(batch_size.eval())
                batch_xs, batch_ys = get_batch_feed(X_train, Y_train, batch_size.eval())
                sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys, phase_train: False})
                # check_point mdl
                if i % arg.report_error_freq == 0:
                    sess.run(step.assign(i))
                    #
                    train_error = sess.run(fetches=loss, feed_dict={x: X_train, y_: Y_train, phase_train: False})
                    cv_error, test_error = -1, -1 # dummy values so that reading data to form plots is easier
                    if arg.collect_generalization:
                        cv_error = sess.run(fetches=loss, feed_dict={x: X_cv, y_: Y_cv, phase_train: False})
                        test_error = sess.run(fetches=loss, feed_dict={x: X_test, y_: Y_test, phase_train: False})
                    if arg.display_training:
                        print( 'step %d, train error: %s | batch_size(step.eval(),arg.batch_size): %s,%s log_learning_rate: %s | mdl %s '%(i,train_error,batch_size_eval,arg.batch_size,arg.log_learning_rate,arg.mdl) )
                    # write files
                    writer.writerow({'train_error':train_error,'cv_error':cv_error,'test_error':test_error})
                # save checkpoint
                if arg.save_checkpoints:
                    if i % arg.save_ckpt_freq == 0:
                        #print('>>>>>>>>>>CKPT',i,arg.save_ckpt_freq)
                        saver.save(sess=sess,save_path=arg.path_to_ckpt+arg.hp_folder_for_ckpt+arg.prefix_ckpt)
                # save last model
                if arg.save_last_mdl:
                    saver.save(sess=sess,save_path=arg.path_to_ckpt+arg.hp_folder_for_ckpt+arg.prefix_ckpt)
            # evaluate
            print('Final Test Acc/error: ', sess.run(fetches=accuracy, feed_dict={x: X_test, y_: Y_test}))
            seconds = (time.time() - start_time)
            minutes = seconds/ 60
            hours = minutes/ 60
            print("--- %s seconds ---" % seconds )
            print("--- %s minutes ---" % minutes )
            print("--- %s hours ---" % hours )
            #arg.seconds, arg.minutes, arg.hours = seconds, minutes, hours

#def set_random_seed():


#

def restore_hps(arg):
    '''
    note: the hps are only restored if there is some tensorflow ckpt, meaning that
    some iteration has been saved/checkpointed. Thus, it means that the hyper params that were saved
    correspond to the hyper params for the checkpointed model. Why? Hyper params
    for a model that has not started are always saved (and overwrite old ones). Thus, if there are no tf
    ckpts it means the restored flag is not set and thus it can't run old hyper params because arg.restore is not true.
    However, if the restore flag is set to true then some iteration must have been ran.
    Thus, the restore flag is true and the hyper params for that run will be used.
    '''
    with open(arg.path_to_hp+arg.json_hp_filename, 'r') as f:
        hps = json.load(f)
    arg = maps.NamedDict(hps['arg_dict'])
    return arg

def save_hps(arg):
    '''
    '''
    # TODO: what does json dump do if the file already exists? we want it to just overwrite it
    with open(arg.path_to_hp+arg.json_hp_filename, 'w+') as f:
        # get arguments to this hp run
        arg_dict = dict( get_remove_functions_from_dict(arg) )
        #
        hps = {'arg_dict':arg_dict}
        json.dump(hps,f,indent=2, separators=(',', ': '))
