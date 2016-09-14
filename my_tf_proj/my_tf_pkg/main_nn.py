import tensorflow as tf

from sklearn import preprocessing
import numpy as np

import shutil
import subprocess
import json
import sys
import datetime
import os
import pdb
import ast
import pickle

import my_tf_pkg as mtf
import time

# tensorboard --logdir=/tmp/mdl_logs

def any_is_NaN(*args):
    is_any_nan = False
    for val in args:
        is_any_nan = np.isnan(val) or is_any_nan
    return is_any_nan

def get_remove_functions_from_dict(arg_dict):
    '''
        Removes functions from dictionary and returns modified dictionary
    '''
    keys_to_delete = []
    for key,value in arg_dict.items():
        if hasattr(value, '__call__'):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del arg_dict[key]
    return arg_dict

def randomnes():
    # randomness TODO
    #tf_rand_seed = int(os.urandom(32).encode('hex'), 16)
    #tf.set_random_seed(tf_rand_seed)
    return

def set_tensorboard(arg):
    # TODO fix and use official way to do this
    print('use_tensorboard', arg.use_tensorboard)
    arg.tensorboard_data_dump_train = '/tmp/mdl_logs/train' #note these names are always saved in results even if its not done
    arg.tensorboard_data_dump_test = '/tmp/mdl_logs/test' #note these names are always saved in results even if its not done
    if arg.use_tensorboard:
        print( '==> tensorboard_data_dump_train: ', arg.tensorboard_data_dump_train )
        print( '==> tensorboard_data_dump_test: ', arg.tensorboard_data_dump_test )
        print( 'mdl_save',arg.mdl_save )
        mtf.make_and_check_dir(path=arg.tensorboard_data_dump_train)
        mtf.make_and_check_dir(path=arg.tensorboard_data_dump_test)
        # delete contents of tensorboard dir
        shutil.rmtree(arg.tensorboard_data_dump_train)
        shutil.rmtree(arg.tensorboard_data_dump_test)
    return

def set_experiment_folders(arg):
    '''
        Way to do experiment:
        - goes to location arg.experiment_root_dir usually = ../../TASK_DATA_NAME
        - goes to arg.experiment_name usually indicates which experiment we want to run.
        for example if want to test 10 centers we do NN_10 as the folder name to hold the different runs
        - then for each hp it saves it in ../../TASK_DATA_NAME/NN_10/hp_name
        with hp_name being some file name for those hyper_params
    '''
    ## directory structure for collecting data for experiments
    #path_root = '%s/%s'%(arg.experiment_root_dir,arg.experiment_name)
    path_root = arg.get_path_root(arg)
    print('path_root: ', path_root)
    #
    arg.date = datetime.date.today().strftime("%B %d").replace (" ", "_")
    current_experiment_folder = '/%s_j%s'%(arg.date,arg.job_name)
    path = path_root+current_experiment_folder
    #
    errors_pretty = '/errors_file_%s_slurm_sj%s.txt'%(arg.date,arg.slurm_array_task_id)
    #
    mdl_dir ='/mdls_%s_slurm_sj%s'%(arg.date,arg.slurm_array_task_id)
    #
    json_file = '/json_%s_slurm_array_id%s_jobid_%s'%(arg.date, arg.slurm_array_task_id, arg.slurm_jobid)
    # try to make directory, if it exists do NOP
    mtf.make_and_check_dir(path=path)
    mtf.make_and_check_dir(path=path+mdl_dir)
    return path, errors_pretty, mdl_dir, json_file

def main_nn(arg):
    print('Running main')
    print('--==>', dict(arg) )
    results = {'train_errors':[], 'cv_errors':[],'test_errors':[]}

    path, errors_pretty, mdl_dir, json_file = set_experiment_folders(arg)
    set_tensorboard(arg)

    ## Data sets and task
    print( '----====> TASK NAME: %s' % arg.task_name )
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(arg.task_name)
    if arg.data_normalize == 'normalize_input':
        X_train, X_cv, X_test = preprocessing.scale(X_train), preprocessing.scale(X_cv), preprocessing.scale(X_test)

    (N_train,D) = X_train.shape
    (N_test,D_out) = Y_test.shape
    print( '(N_train,D) = ', (N_train,D) )
    print( '(N_test,D_out) = ', (N_test,D_out) )

    ##
    phase_train = tf.placeholder(tf.bool, name='phase_train') if arg.bn else  None

    arg.steps = arg.get_steps(arg)
    arg.M = arg.get_batch_size(arg)

    arg.log_learning_rate = arg.get_log_learning_rate(arg)
    arg.starter_learning_rate = arg.get_start_learning_rate(arg)
    print( '++> starter_learning_rate ', arg.starter_learning_rate )

    ## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    arg.decay_rate = arg.get_decay_rate(arg)
    arg.decay_steps = arg.get_decay_steps(arg)

    if arg.optimization_alg == 'GD':
        pass
    elif arg.optimization_alg=='Momentum':
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
    else:
        pass

    ##############################
    # if task_name == 'task_MNIST_flat_auto_encoder':
    #     PCA_errors = {12:24.8254684915, 48:9.60052317906, 96:4.72118325768}
    #     if len(units_list) == 1:
    #         k = units_list[0]
    #     else:
    #         k = units_list[0] * len(units_list)
    #     if not k in PCA_errors.keys():
    #         print( 'COMPUTING PCA... k = ', k)
    #         X_reconstruct_pca, _, _ = mtf. get_reconstruction_pca(X_train,k=units_list[0])
    #         pca_error = mtf.report_l2_loss(Y=X_train,Y_pred=X_reconstruct_pca)
    #         PCA_errors[k] = pca_error
    #     else:
    #         pca_error = PCA_errors[k]
    #     print( '*************> PCA error: ', pca_error)
    # else:
    #     pca_error = None
    #     rbf_error = None
    #
    # hbf1_error = None
    # if model == 'hbf':
    #     #error, Y_pred, Kern, C, subsampled_data_points = report_RBF_error_from_data(X_train, dims, stddev)
    #     if len(units_list) > 1:
    #         k = units_list[0]*len(units_list)
    #         print( 'RBF units = ', k)
    #         nb_units = [None, k]
    #         rbf_error, _, _, _, _ = mtf.report_RBF_error_from_data(X_train, X_train, nb_units, S_init[1])
    #         print( rbf_error)
    #         hbf1={12:26.7595}
    #         if k in hbf1.keys():
    #             hbf1_error = hbf1[k]
    #     else:
    #         nb_units = dims
    #         rbf_error, _, _, _, _ = mtf.report_RBF_error_from_data(X_train, X_train, nb_units, S_init[1])
    ##

    pca_error = None
    rbf_error = None
    hbf1_error = None
    ## Make Model
    if arg.mdl == 'standard_nn':
        arg.dims = [D]+arg.units+[D_out]
        arg.mu_init_list = arg.get_W_mu_init(arg)
        arg.std_init_list = arg.get_W_std_init(arg)

        arg.b_init = arg.get_b_init(arg)
        float_type = tf.float64
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D

        nb_layers = len(arg.dims)-1
        nb_hidden_layers = nb_layers-1
        (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=arg.init_type,dims=arg.dims,mu=arg.mu_init_list,std=arg.std_init_list,b_init=arg.b_init, X_train=X_train, Y_train=Y_train)
        with tf.name_scope("standardNN") as scope:
            mdl = mtf.build_standard_NN(arg, x,arg.dims,(None,inits_W,inits_b),phase_train,arg.trainable_bn)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
        inits_S = inits_b
    elif arg.mdl == 'hbf':
        arg.dims = [D]+arg.units+[D_out]
        trainable_S = True if (arg.trainable_S=='train_S') else False
        arg.b_init = arg.get_b_init(arg)
        arg.S_init = arg.b_init
        float_type = tf.float64
        #arg.mu , arg.std = arg.get_W_mu_init(arg), arg.get_W_std_init(arg)
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D
        (inits_C,inits_W,inits_S,rbf_error) = mtf.get_initilizations_HBF(init_type=arg.init_type,dims=arg.dims,mu=arg.mu,std=arg.std,b_init=arg.b_init,S_init=arg.S_init, X_train=X_train, Y_train=Y_train, train_S_type=arg.train_S_type)
        #print(inits_W)
        nb_layers = len(arg.dims)-1
        nb_hidden_layers = nb_layers-1
        with tf.name_scope("HBF") as scope:
            mdl = mtf.build_HBF2(x,arg.dims,(inits_C,inits_W,inits_S),phase_train,arg.trainable_bn,trainable_S)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
    elif arg.mdl == 'binary_tree_4D_conv':
        pass
        # print( 'binary_tree_4D')
        # #tensorboard_data_dump = '/tmp/hbf_logs'
        # inits_S = None
        # pca_error = None
        # rbf_error = None
        # float_type = tf.float32
        # # things that need reshaping
        # N_cv = X_cv.shape[0]
        # N_test = X_test.shape[0]
        # #
        # X_train = X_train.reshape(N_train,1,D,1)
        # X_cv = X_cv.reshape(N_cv,1,D,1)
        # X_test = X_test.reshape(N_test,1,D,1)
        # x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        # #
        # arg.filter_size = 2 #fixed for Binary Tree BT
        # nb_filters = arg.nb_filters
        # mean, stddev = arg.mu, arg.std
        # stddev = float( np.random.uniform(low=0.001, high=stddev) )
        # print( 'stddev', stddev)
        # x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        # with tf.name_scope("build_binary_model") as scope:
        #     mdl = mtf.build_binary_tree(x,arg.filter_size,nb_filters,mean,stddev,stride_convd1=2,phase_train=phase_train,trainable_bn=arg.trainable_bn)
        #
        # arg.dims = [D]+[nb_filters]+[D_out]
    elif arg.mdl == 'binary_tree_4D_conv_hidden_layer':
        print( 'binary_tree_4D' )
        inits_S = None
        pca_error, rbf_error = None, None
        float_type = tf.float32
        # Data sizes needed for reshaping
        N_cv, N_test = X_cv.shape[0], X_test.shape[0]
        # reshape data sets
        X_train, X_cv, X_test = X_train.reshape(N_train,1,D,1), X_cv.reshape(N_cv,1,D,1), X_test.reshape(N_test,1,D,1)
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        #
        arg.stride_convd1, arg.filter_size = 2, 2 #fixed for Binary Tree BT
        arg.mean, arg.stddev = arg.get_W_mu_init(arg), arg.get_W_std_init(arg)
        with tf.name_scope("build_binary_model") as scope:
            mdl = mtf.build_binary_tree_4D_hidden_layer(x,arg,phase_train=phase_train)
        arg.dims = [D]+[arg.nb_filters]+[arg.nb_final_hidden_units]+[D_out]
    elif arg.mdl == 'binary_tree_8D_conv':
        pass
        # print( 'binary_tree_8D_conv')
        # #tensorboard_data_dump = '/tmp/hbf_logs'
        # inits_S = None
        # pca_error = None
        # rbf_error = None
        # float_type = tf.float32
        # # things that need reshaping
        # N_cv = X_cv.shape[0]
        # N_test = X_test.shape[0]
        # #
        # X_train = X_train.reshape(N_train,1,D,1)
        # X_cv = X_cv.reshape(N_cv,1,D,1)
        # X_test = X_test.reshape(N_test,1,D,1)
        # x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        # #
        # filter_size = 2 #fixed for Binary Tree BT
        # nb_filters1,nb_filters2 = arg.nb_filters
        # arg.mean = arg.get_W_mu_init(arg)
        # mean1,mean2,mean3 = arg.mean
        # arg.stddev = arg.get_W_std_init(arg)
        # stddev1,stddev2,stddev3 = arg.stddev
        # x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        # with tf.name_scope("binary_tree_D8") as scope:
        #     mdl = mtf.build_binary_tree_8D(x,nb_filters1,nb_filters2,mean1,stddev1,mean2,stddev2,mean3,stddev3,stride_conv1=2)
        # #
        # arg.dims = [D]+arg.nb_filters+[D_out]

    ## Output and Loss
    y = mdl
    y_ = tf.placeholder(float_type, shape=[None, D_out]) # (M x D)
    with tf.name_scope("L2_loss") as scope:
        l2_loss = tf.reduce_sum( tf.reduce_mean(tf.square(y_-y), 0) )
        #l2_loss = (2.0/N_train)*tf.nn.l2_loss(y_-y)
        #l2_loss = tf.reduce_mean(tf.square(y_-y))

    ##

    with tf.name_scope("train") as scope:
        # If the argument staircase is True, then global_step / decay_steps is an integer division and the decayed earning rate follows a staircase function.
        ## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=arg.starter_learning_rate, global_step=global_step,decay_steps=arg.decay_steps, decay_rate=arg.decay_rate, staircase=arg.staircase)
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

    ## TODO
    if arg.re_train == 're_train' and arg.task_name == 'hrushikesh':
        print( 'task_name: ', task_name)
        print( 're_train: ', re_train)
        var_list = [v for v in tf.all_variables() if v.name == 'C:0']
        #train_step = opt.minimize(l2_loss, var_list=var_list)
    else:
        train_step = opt.minimize(l2_loss, global_step=global_step)

    ##
    with tf.name_scope('learning_rate'):
        learning_rate_scalar_summary = tf.scalar_summary("learning_rate", learning_rate)

    with tf.name_scope("l2_loss") as scope:
        ls_scalar_summary = tf.scalar_summary("l2_loss", l2_loss)

    if arg.task_name == 'task_MNIST_flat_auto_encoder':
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

    def register_all_variables_and_grads(y):
        all_vars = tf.all_variables()
        grad_vars = opt.compute_gradients(y,all_vars) #[ (gradient,variable) ]
        for (dldw,v) in grad_vars:
            if dldw != None:
                prefix_name = 'derivative_'+v.name
                suffix_text = 'dJd'+v.name
                #mtf.put_summaries(var=tf.sqrt( tf.reduce_sum(tf.square(dldw)) ),prefix_name=prefix_name,suffix_text=suffix_text)
                mtf.put_summaries(var=tf.abs(dldw),prefix_name=prefix_name,suffix_text='_abs_'+suffix_text)
                tf.histogram_summary('hist'+prefix_name, dldw)

    register_all_variables_and_grads(y)
    ## TRAIN
    if phase_train is not None:
        #DO BN
        feed_dict_train = {x:X_train, y_:Y_train, phase_train: False}
        feed_dict_cv = {x:X_cv, y_:Y_cv, phase_train: False}
        feed_dict_test = {x:X_test, y_:Y_test, phase_train: False}
    else:
        #Don't do BN
        feed_dict_train = {x:X_train, y_:Y_train}
        feed_dict_cv = {x:X_cv, y_:Y_cv}
        feed_dict_test = {x:X_test, y_:Y_test}

    def get_batch_feed(X, Y, M, phase_train):
        mini_batch_indices = np.random.randint(M,size=M)
        Xminibatch =  X[mini_batch_indices,:] # ( M x D^(0) )
        Yminibatch = Y[mini_batch_indices,:] # ( M x D^(L) )
        if phase_train is not None:
            #DO BN
            feed_dict = {x: Xminibatch, y_: Yminibatch, phase_train: True}
        else:
            #Don't do BN
            feed_dict = {x: Xminibatch, y_: Yminibatch}
        return feed_dict

    def print_messages(*args):
        for i, msg in enumerate(args):
            print('>%s'%msg, flush=True)

    if arg.use_tensorboard:
        if tf.gfile.Exists('/tmp/mdl_logs'):
          tf.gfile.DeleteRecursively('/tmp/mdl_logs')
        tf.gfile.MakeDirs('/tmp/mdl_logs')

    tf.add_check_numerics_ops()

    # Add ops to save and restore all the variables.
    if arg.mdl_save:
        saver = tf.train.Saver(max_to_keep=arg.max_to_keep)
    start_time = time.time()
    print()
    #file_for_error = './ray_error_file.txt'
    if arg.save_config_args:
        arg_dict = dict(arg).copy()
        arg_dict = get_remove_functions_from_dict(arg_dict)
        pickle.dump( arg_dict, open( "pickle-slurm-%s_%s.p"%(arg.slurm_jobid,arg.slurm_array_task_id) , "wb" ) )
        #with open('json-slurm-%s_%s.json', 'w+') as f_json:
        #    json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
    with open(path+errors_pretty, 'w+') as f_err_msgs:
    #with open(file_for_error, 'w+') as f_err_msgs:
        with tf.Session() as sess:
            ## prepare writers and fetches
            if arg.use_tensorboard:
                merged = tf.merge_all_summaries()
                #writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)
                train_writer = tf.train.SummaryWriter(arg.tensorboard_data_dump_train, sess.graph)
                test_writer = tf.train.SummaryWriter(arg.tensorboard_data_dump_test, sess.graph)
                ##
                fetches_train = [merged, l2_loss]
                fetches_cv = l2_loss
                fetches_test = [merged, l2_loss]
            else:
                fetches_train = l2_loss
                fetches_cv = l2_loss
                fetches_test = l2_loss

            sess.run( tf.initialize_all_variables() )
            for i in range(arg.steps):
                ## Create fake data for y = W.x + b where W = 2, b = 0
                #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
                feed_dict_batch = get_batch_feed(X_train, Y_train, arg.M, phase_train)
                ## Train
                if i%arg.report_error_freq == 0:
                    if arg.use_tensorboard:
                        (summary_str_train,train_error) = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
                        cv_error = sess.run(fetches=fetches_cv, feed_dict=feed_dict_cv)
                        (summary_str_test,test_error) = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

                        train_writer.add_summary(summary_str_train, i)
                        test_writer.add_summary(summary_str_test, i)
                    else:
                        train_error = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
                        cv_error = sess.run(fetches=fetches_cv, feed_dict=feed_dict_cv)
                        test_error = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

                    current_learning_rate = sess.run(fetches=learning_rate)
                    loss_msg = "=> Mdl*%s*-units%s, task: %s, step %d/%d, train err %g, cv err: %g test err %g"%(arg.mdl,arg.dims,arg.task_name,i,arg.steps,train_error,cv_error,test_error)
                    mdl_info_msg = "Opt:%s, BN %s, BN_trainable: %s After%d/%d iteration,Init: %s, current_learning_rate %s, M %s, decay_rate %s, decay_steps %s" % (arg.optimization_alg,arg.bn,arg.trainable_bn,i,arg.steps,arg.init_type,current_learning_rate,arg.M,arg.decay_rate,arg.decay_steps)
                    errors_to_beat = 'BEAT: hbf1_error: %s RBF error: %s PCA error: %s '%(hbf1_error, rbf_error,pca_error)

                    print_messages(loss_msg, mdl_info_msg, errors_to_beat)
                    print('S: ', inits_S, flush=True)
                    print()

                    # store results
                    results['train_errors'].append( float(train_error) )
                    results['cv_errors'].append( float(cv_error) )
                    results['test_errors'].append( float(test_error) )
                    # write errors to pretty print
                    f_err_msgs.write(loss_msg)
                    f_err_msgs.write(mdl_info_msg)
                    if any_is_NaN(train_error,cv_error,test_error):
                        # if its a nan make sure to stop script
                        print('nan_found')
                        break
                    if arg.mdl_save:
                        save_path = saver.save(sess, path+mdl_dir+'/model.ckpt',global_step=i)
                if arg.use_tensorboard:
                    sess.run(fetches=[merged,train_step], feed_dict=feed_dict_batch) #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                else:
                    sess.run(fetches=train_step, feed_dict=feed_dict_batch) #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    best_train, best_cv, best_test =  mtf.get_errors_from(results)
    results['best_train'], results['best_cv'], results['best_test'] = best_train, best_cv, best_test
    print('End of main')

    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    results['git_hash'] = str(git_hash)
    #results['tf_rand_seed'] = tf_rand_seed
    #
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    ## dump results to JSON
    results['seconds'] = seconds
    results['minutes'] = minutes
    results['hours'] = hours
    #print results
    #results['arg'] = arg
    arg_dict = dict(arg)
    arg_dict = get_remove_functions_from_dict(arg_dict)
    results['arg_dict'] = arg_dict
    with open(path+json_file, 'w+') as f_json:
        print('Writing Json')
        print('path+json_file', path+json_file)
        json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
    print( '\a') #makes beep
    #print(results)
    print( 'best results: train, cv, test: ', best_train, best_cv, best_test )
