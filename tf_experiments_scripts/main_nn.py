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

import my_tf_pkg as mtf
import time

def randomnes():
    # randomness
    #tf_rand_seed = int(os.urandom(32).encode('hex'), 16)
    #tf.set_random_seed(tf_rand_seed)
    return

def set_tensorboard(arg):
    #
    print('use_tensorboard', arg.use_tensorboard)
    tensorboard_data_dump_train = '/tmp/mdl_logs/train' #note these names are always saved in results even if its not done
    tensorboard_data_dump_test = '/tmp/mdl_logs/test' #note these names are always saved in results even if its not done
    if arg.use_tensorboard:
        print( '==> tensorboard_data_dump_train: ', tensorboard_data_dump_train )
        print( '==> tensorboard_data_dump_test: ', tensorboard_data_dump_test )
        print( 'mdl_save',arg.mdl_save )
        mtf.make_and_check_dir(path=tensorboard_data_dump_train)
        mtf.make_and_check_dir(path=tensorboard_data_dump_test)
        # delete contents of tensorboard dir
        shutil.rmtree(tensorboard_data_dump_train)
        shutil.rmtree(tensorboard_data_dump_test)
    return

def set_experiment_folders(arg):
    ## directory structure for collecting data for experiments
    path_root = '../../%s/%s'%(arg.experiment_root_dir,arg.experiment_name)
    #
    date = datetime.date.today().strftime("%B %d").replace (" ", "_")
    current_experiment_folder = '/%s_j%s'%(date,arg.job_name)
    path = path_root+current_experiment_folder
    #
    errors_pretty = '/errors_file_%s_slurm_sj%s.txt'%(date,arg.slurm_array_task_id)
    #
    mdl_dir ='/mdls_%s_slurm_sj%s'%(date,arg.slurm_array_task_id)
    #
    json_file = '/json_%s_slurm_array_id%s_jobid_%s'%(date, arg.slurm_array_task_id, arg.slurm_jobid)
    # try to make directory, if it exists do NOP
    mtf.make_and_check_dir(path=path)
    mtf.make_and_check_dir(path=path+mdl_dir)
    return path, errors_pretty, mdl_dir, json_file

def main(arg):
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

    steps = np.random.randint(low=arg.steps_low ,high=arg.steps_high)
    M = np.random.randint(low=arg.M_low , high=arg.M_high)
    arg.M = M

    log_learning_rate = np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
    starter_learning_rate = 10**log_learning_rate
    print( '++> starter_learning_rate ', starter_learning_rate )

    ## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    decay_rate = np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)
    arg.decay_steps_high
    decay_steps = np.random.randint(low=arg.decay_steps_low(arg), high=arg.decay_steps_high(arg) )
    staircase = arg.staircase

    if arg.optimization_alg == 'GD':
        pass
    elif arg.optimization_alg=='Momentum':
        use_nesterov = arg.use_nesterov
        momentum=np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
        results['momentum']=float(momentum)
    elif arg.optimization_alg == 'Adadelta':
        rho=np.random.uniform(low=arg.rho_low,high=arg.rho_high)
        results['rho']=float(rho)
    elif arg.optimization_alg == 'Adagrad':
        #only has learning rate
        pass
    elif arg.optimization_alg == 'Adam':
        beta1 = arg.get_beta1(arg)
        beta2 = arg.get_beta2(arg)
        results['beta1']=float(beta1)
        results['beta2']=float(beta2)
    elif arg.optimization_alg == 'RMSProp':
        decay = np.random.uniform(low=arg.decay_loc,high=arg.decay_high)
        momentum = np.random.uniform(low=arg.momentum_low,high=arg.momentum_high)
        results['decay']=float(decay)
        results['momentum']=float(momentum)
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

    ## Make Model
    if arg.mdl == 'standard_nn':
        arg.dims = [D]+arg.units+[D_out]
        mu = arg.W_mu_init(arg)
        std = arg.W_std_init(arg)

        b_init = arg.b_init()

        rbf_error = None
        float_type = tf.float64
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D

        nb_layers = len(arg.dims)-1
        nb_hidden_layers = nb_layers-1
        (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=arg.init_type,dims=arg.dims,mu=arg.mu,std=arg.std,b_init=b_init, X_train=X_train, Y_train=Y_train)
        with tf.name_scope("standardNN") as scope:
            mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train,trainable_bn)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
        inits_S = inits_b
    elif arg.mdl == 'hbf':
        trainable_S = True if (arg.trainable_S=='train_S') else False
        #tensorboard_data_dump = '/tmp/hbf_logs'
        float_type = tf.float64
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D
        (inits_C,inits_W,inits_S,rbf_error) = mtf.get_initilizations_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train, train_S_type=train_S_type)
        print(inits_W)
        with tf.name_scope("HBF") as scope:
            mdl = mtf.build_HBF2(x,dims,(inits_C,inits_W,inits_S),phase_train,trainable_bn,trainable_S)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
    elif arg.mdl == 'binary_tree_4D_conv':
        print( 'binary_tree_4D')
        #tensorboard_data_dump = '/tmp/hbf_logs'
        inits_S = None
        pca_error = None
        rbf_error = None
        float_type = tf.float32
        # things that need reshaping
        N_cv = X_cv.shape[0]
        N_test = X_test.shape[0]
        #
        X_train = X_train.reshape(N_train,1,D,1)
        X_cv = X_cv.reshape(N_cv,1,D,1)
        X_test = X_test.reshape(N_test,1,D,1)
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        #
        filter_size = 2 #fixed for Binary Tree BT
        #nb_filters = nb_filters
        mean, stddev = bn_tree_init_stats
        stddev = float( np.random.uniform(low=0.001, high=stddev) )
        print( 'stddev', stddev)
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        with tf.name_scope("build_binary_model") as scope:
            mdl = mtf.build_binary_tree(x,filter_size,nb_filters,mean,stddev,stride_convd1=2,phase_train=phase_train,trainable_bn=trainable_bn)
        #
        dims = [D]+[nb_filters]+[D_out]
        results['nb_filters'] = nb_filters
    elif arg.mdl == 'binary_tree_D8':
        #tensorboard_data_dump = '/tmp/hbf_logs'
        inits_S = None
        pca_error = None
        rbf_error = None
        float_type = tf.float32
        # things that need reshaping
        N_cv = X_cv.shape[0]
        N_test = X_test.shape[0]
        #
        X_train = X_train.reshape(N_train,1,D,1)
        #Y_train = Y_train.reshape(N_train,1,D,1)
        X_cv = X_cv.reshape(N_cv,1,D,1)
        #Y_cv = Y_cv.reshape(N_cv,1,D,1)
        X_test = X_test.reshape(N_test,1,D,1)
        #Y_test = Y_test.reshape(N_test,1,D,1)
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        #
        filter_size = 2 #fixed for Binary Tree BT
        nb_filters1,nb_filters2 = nb_filters
        mean1,stddev1,mean2,stddev2,mean3,stddev3 = bn_tree_init_stats
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        with tf.name_scope("binary_tree_D8") as scope:
            mdl = mtf.build_binary_tree_8D(x,nb_filters1,nb_filters2,mean1,stddev1,mean2,stddev2,mean3,stddev3,stride_conv1=2)
        #
        dims = [D]+nb_filters+[D_out]
        results['nb_filters'] = nb_filters

    ## Output and Loss
    y = mdl
    y_ = tf.placeholder(float_type, shape=[None, D_out]) # (M x D)
    with tf.name_scope("L2_loss") as scope:
        l2_loss = tf.reduce_sum( tf.reduce_mean(tf.square(y_-y), 0) )
        #l2_loss = (2.0/N_train)*tf.nn.l2_loss(y_-y)
        #l2_loss = tf.reduce_mean(tf.square(y_-y))

    ##

    with tf.name_scope("train") as scope:
        # starter_learning_rate = 0.0000001
        # decay_rate = 0.9
        # decay_steps = 100
        # staircase = True
        # decay_steps = 10000000
        # staircase = False
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

        # Passing global_step to minimize() will increment it at each step.
        if optimization_alg == 'GD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimization_alg == 'Momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum,use_nesterov=use_nesterov)
        elif optimization_alg == 'Adadelta':
            tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=rho, epsilon=1e-08, use_locking=False, name='Adadelta')
        elif optimization_alg == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08, name='Adam')
        elif optimization_alg == 'Adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimization_alg == 'RMSProp':
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=1e-10, name='RMSProp')

    ##
    if re_train == 're_train' and task_name == 'hrushikesh':
        print( 'task_name: ', task_name)
        print( 're_train: ', re_train)
        var_list = [v for v in tf.all_variables() if v.name == 'C:0']
        #train_step = opt.minimize(l2_loss, var_list=var_list)
    else:
        train_step = opt.minimize(l2_loss, global_step=global_step)

    ##
    with tf.name_scope("l2_loss") as scope:
        ls_scalar_summary = tf.scalar_summary("l2_loss", l2_loss)

    if task_name == 'task_MNIST_flat_auto_encoder':
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
            print('>',msg)

    if use_tensorboard:
        if tf.gfile.Exists('/tmp/mdl_logs'):
          tf.gfile.DeleteRecursively('/tmp/mdl_logs')
        tf.gfile.MakeDirs('/tmp/mdl_logs')

    tf.add_check_numerics_ops()

    # Add ops to save and restore all the variables.
    if arg.mdl_save:
        saver = tf.train.Saver(max_to_keep=arg.max_to_keep)
    start_time = time.time()
    file_for_error = './ray_error_file.txt'
    with open(path+errors_pretty, 'w+') as f_err_msgs:
    #with open(file_for_error, 'w+') as f_err_msgs:
        with tf.Session() as sess:
            ## prepare writers and fetches
            if use_tensorboard:
                merged = tf.merge_all_summaries()
                #writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)
                train_writer = tf.train.SummaryWriter(tensorboard_data_dump_train, sess.graph)
                test_writer = tf.train.SummaryWriter(tensorboard_data_dump_test, sess.graph)
                ##
                fetches_train = [merged, l2_loss]
                fetches_cv = l2_loss
                fetches_test = [merged, l2_loss]
            else:
                fetches_train = l2_loss
                fetches_cv = l2_loss
                fetches_test = l2_loss

            sess.run( tf.initialize_all_variables() )
            for i in range(steps):
                ## Create fake data for y = W.x + b where W = 2, b = 0
                #(batch_xs, batch_ys) = get_batch_feed(X_train, Y_train, M, phase_train)
                feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
                ## Train
                if i%report_error_freq == 0:
                    if use_tensorboard:
                        (summary_str_train,train_error) = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
                        cv_error = sess.run(fetches=fetches_cv, feed_dict=feed_dict_cv)
                        (summary_str_test,test_error) = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

                        train_writer.add_summary(summary_str_train, i)
                        test_writer.add_summary(summary_str_test, i)
                    else:
                        train_error = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
                        cv_error = sess.run(fetches=fetches_cv, feed_dict=feed_dict_cv)
                        test_error = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

                    loss_msg = "Mdl*%s%s*-units%s, task: %s, step %d/%d, train err %g, cv err: %g test err %g"%(model,nb_hidden_layers,dims,task_name,i,steps,train_error,cv_error,test_error)
                    mdl_info_msg = "Opt:%s, BN %s, BN_trainable: %s After%d/%d iteration,Init: %s" % (optimization_alg,bn,trainable_bn,i,steps,init_type)
                    errors_to_beat = 'BEAT: hbf1_error: %s RBF error: %s PCA error: %s '%(hbf1_error, rbf_error,pca_error)
                    print_messages(loss_msg, mdl_info_msg, errors_to_beat)
                    #sys.stdout.flush()
                    loss_msg+="\n"
                    mdl_info_msg+="\n"
                    errors_to_beat+="\n"

                    print( 'S: ', inits_S)
                    # store results
                    #print type(train_error)
                    results['train_errors'].append( float(train_error) )
                    #print type(cv_error)
                    results['cv_errors'].append( float(cv_error) )
                    #print type(test_error)
                    results['test_errors'].append( float(test_error) )
                    # write errors to pretty print
                    f_err_msgs.write(loss_msg)
                    f_err_msgs.write(mdl_info_msg)
                    # save mdl
                    if mdl_save:
                        save_path = saver.save(sess, path+mdl_dir+'/model.ckpt',global_step=i)
                if use_tensorboard:
                    sess.run(fetches=[merged,train_step], feed_dict=feed_dict_batch) #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                else:
                    sess.run(fetches=train_step, feed_dict=feed_dict_batch) #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    mtf.load_results_dic(results,git_hash=git_hash,dims=dims,mu=mu,std=std,init_constant=init_constant,b_init=b_init,S_init=S_init,\
        init_type=init_type,model=model,bn=bn,path=path,\
        tensorboard_data_dump_test=tensorboard_data_dump_test,tensorboard_data_dump_train=tensorboard_data_dump_train,\
        report_error_freq=report_error_freq,steps=steps,M=M,optimization_alg=optimization_alg,\
        starter_learning_rate=starter_learning_rate,decay_rate=decay_rate,staircase=staircase)

    ##
    results['job_name'] = job_name
    results['slurm_jobid'] = slurm_jobid
    results['slurm_array_task_id'] = slurm_array_task_id
    #results['tf_rand_seed'] = tf_rand_seed
    results['date'] = date
    results['bn'] = bn
    results['trainable_bn'] = trainable_bn

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
    with open(path+json_file, 'w+') as f_json:
        json.dump(results,f_json,sort_keys=True, indent=2, separators=(',', ': '))
    print( '\a') #makes beep



# low_const, high_const = 0.4, 1.0
# #init_constant = np.random.uniform(low=low_const, high=high_const)
# #b_init = list(np.random.uniform(low=low_const, high=high_const,size=len(dims)))
# init_constant = 0.4177551
# #b_init = len(dims)*[init_constant]
# #[0.6374998052942504, 0.6374998052942504, 0.6374998052942504, 0.6374998052942504]
# b_init = [None, init_constant, np.random.uniform(low=1,high=2.5)]

if __name__ == '__main__':
    print( 'in __main__')
    print( 'start running main_nn.py')
    main()
    print( 'end running main_nn.py')
