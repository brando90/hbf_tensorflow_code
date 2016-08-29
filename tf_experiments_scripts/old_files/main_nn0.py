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

def main():
    #print 'print sys.argv =',sys.argv
    #print 'len(sys.argv) =',len(sys.argv)

    def init_norm(loc,scale,upper_threshold,lower_threshold):
        init_constant = np.random.normal(loc=loc,scale=scale)
        for i, item in enumerate(init_constant):
            if item > upper_threshold:
                init_constant[i] = upper_threshold
            elif item < lower_threshold:
                init_constant[i] = lower_threshold
        return

    def get_init_b(argv_init_S,dims):
        #parse argv input
        type_S, arg_S = argv_init_S.split('-')
        print '++> type_S ', type_S
        print '++> arg_S ', arg_S
        # process it and choose constant list:
        if type_S == 'all_same_const':
            # arg_S = init_constant
            init_constant = float(arg_S)
            b_init = len(dims)*[init_constant]
        elif type_S == 'first_constant_rest_specific_consts':
            # arg_S = init_constant,const_1,...const_l,...,const_L]
            b_init = ast.literal_eval(arg_S)
            b_init = [None] + b_init
        elif type_S == 'first_constant_rest_uniform_random':
            # arg_S = [init_constant,(low_1,high_1),...(low_l,high_l),...,(low_L,high_L)]
            params_S = ast.literal_eval(arg_S)
            b_init = [None, params_S[0]]
            for l in range(1,len(params_S)):
                #b_init.append( np.random.uniform(low=1,high=2.5) )
                low, high = params_S[l]
                print '###########> params_S', params_S[l]
                b_init.append( np.random.uniform(low=low,high=high) )
        elif type_S == 'first_rand_same_uniform_rest_uniform_random':
            # arg_S = [(low_1,high_1)  ,  (low_1,high_1),...(low_l,high_l),...,(low_L,high_L)]
            params_S = ast.literal_eval(arg_S)
            low, high = params_S[0]
            b_init = [None, np.random.uniform(low=low,high=high)]
            for l in range(1,len(params_S)):
                #b_init.append( np.random.uniform(low=1,high=2.5) )
                low, high = params_S[l]
                print '###########> params_S', params_S[l]
                b_init.append( np.random.uniform(low=low,high=high) )
        else:
            raise ValueError('Wrong type of b/S init')
        print '++===> S/b_init ', b_init
        return b_init

    re_train = None
    #re_train = 're_train'
    results = {'train_errors':[], 'cv_errors':[],'test_errors':[]}
    # slurm values and ids
    #(experiment_root_dir,slurm_jobid,slurm_array_task_id,job_name,mdl_save,experiment_name,units_list,train_S_type,task_name,bn,trainable_bn,mdl_type,init_type,cluster,data_normalize,trainable_S,argv_init_S,optimization_alg,nb_filters,bn_tree_init_stats) = mtf.process_argv(sys.argv)
    (experiment_root_dir,slurm_jobid,slurm_array_task_id,job_name,mdl_save,experiment_name,units_list,train_S_type,task_name,bn,trainable_bn,mdl_type,init_type,cluster,data_normalize,trainable_S,argv_init_S,optimization_alg,nb_filters,bn_tree_init_stats) = mtf.process_argv('')
    results['task_name'] = task_name
    results['argv_init_S'] = argv_init_S
    results['train_S_type'] = train_S_type
    results['trainable_S'] = trainable_S

    #use_tensorboard = mtf.is_it_tensorboard_run(sys.argv)
    use_tensorboard = False
    #use_tensorboard =  False
    trainable_S = True if (trainable_S=='train_S') else False
    print 'use_tensorboard', use_tensorboard
    date = datetime.date.today().strftime("%B %d").replace (" ", "_")
    print 'experiment_root_dir=%s,slurm_jobid=%s,slurm_array_task_id=%s,job_name=%s'%(experiment_root_dir,slurm_jobid,slurm_array_task_id,job_name)

    # randomness
    tf_rand_seed = int(os.urandom(32).encode('hex'), 16)
    tf.set_random_seed(tf_rand_seed)
    ## directory structure for collecting data for experiments
    path_root = '../../%s/%s'%(experiment_root_dir,experiment_name)
    #
    current_experiment_folder = '/%s_j%s'%(date,job_name)
    path = path_root+current_experiment_folder
    #
    #errors_pretty_dir = '/errors_pretty_dir'
    errors_pretty = '/errors_file_%s_slurm_sj%s.txt'%(date,slurm_array_task_id)
    #
    mdl_dir ='/mdls_%s_slurm_sj%s'%(date,slurm_array_task_id)
    #
    #json_dir = '/results_json_dir'
    json_file = '/json_%s_slurm_array_id%s_jobid_%s'%(date, slurm_array_task_id, slurm_jobid)
    #
    tensorboard_data_dump_train = '/tmp/mdl_logs/train' #note these names are always saved in results even if its not done
    tensorboard_data_dump_test = '/tmp/mdl_logs/test' #note these names are always saved in results even if its not done
    if use_tensorboard:
        print '==> tensorboard_data_dump_train: ', tensorboard_data_dump_train
        print '==> tensorboard_data_dump_test: ', tensorboard_data_dump_test
        print 'mdl_save',mdl_save
        mtf.make_and_check_dir(path=tensorboard_data_dump_train)
        mtf.make_and_check_dir(path=tensorboard_data_dump_test)
        # delete contents of tensorboard dir
        shutil.rmtree(tensorboard_data_dump_train)
        shutil.rmtree(tensorboard_data_dump_test)
    # try to make directory, if it exists do NOP
    mtf.make_and_check_dir(path=path)
    #make_and_check_dir(path=path+json_dir)
    #make_and_check_dir(path=path+errors_pretty_dir)
    mtf.make_and_check_dir(path=path+mdl_dir)
    # JSON results structure
    results_dic = mtf.fill_results_dic_with_np_seed(np_rnd_seed=np.random.get_state(), results=results)

    ## Data sets and task
    print '----====> TASK NAME: %s' % task_name
    (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
    if data_normalize == 'normalize_input':
        X_train, X_cv, X_test = preprocessing.scale(X_train), preprocessing.scale(X_cv), preprocessing.scale(X_test)

    (N_train,D) = X_train.shape
    (N_test,D_out) = Y_test.shape
    print '(N_train,D) = ', (N_train,D)
    print '(N_test,D_out) = ', (N_test,D_out)

    init_constant = -1
    ## HBF/NN params
    print 'CLUSTER: ', cluster
    if cluster == 'OM7':
        dims = [D]+units_list+[D_out]
        mu_init = 0.0
        mu = len(dims)*[mu_init]
        #std_init = 0.1
        std_init = float( np.random.uniform(low=0.001, high=1.0) )
        std = len(dims)*[std_init]

        b_init = get_init_b(argv_init_S,dims)
        # if mdl_type == 'binary_tree' or mdl_type=='standard_nn':
        #     b_init = 0.1
        # else:
        #     b_init = get_init_b(argv_init_S,dims)

        model = mdl_type
        max_to_keep = 1

        phase_train = tf.placeholder(tf.bool, name='phase_train') if bn else  None

        report_error_freq = 30
        #steps = np.random.randint(low=3000,high=6000)
        steps = 30000
        M = np.random.randint(low=500, high=15000)
        #M = 17000 #batch-size
        #M = 5000
        print '++++> M (batch size) :', M

        low_const_learning_rate, high_const_learning_rate = -0.01, -6
        log_learning_rate = np.random.uniform(low=low_const_learning_rate, high=high_const_learning_rate)
        starter_learning_rate = 10**log_learning_rate

        print '++> starter_learning_rate ', starter_learning_rate
        ## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decay_rate = np.random.uniform(low=0.3, high=0.99)
        decay_steps = np.random.randint(low=report_error_freq, high=M)
        staircase = True
        print '++> decay_rate ', decay_rate
        print '++> decay_steps ', decay_steps
        print '++> staircase ', staircase

        if optimization_alg == 'GD':
            pass
        elif optimization_alg=='Momentum':
            #use_nesterov=False
            #momentum = 0.9
            momentum=np.random.uniform(low=0.1, high=0.99)
            results['momentum']=float(momentum)
        elif optimization_alg == 'Adadelta':
            #rho = 0.95
            rho=np.random.uniform(low=0.4, high=0.99)
            results['rho']=float(rho)
        elif optimization_alg == 'Adagrad':
            #only has learning rate
            pass
        elif optimization_alg == 'Adam':
            beta1=0.99 # m = b1m + (1 - b1)m
            beta2=0.999 # v = b2 v + (1 - b2)v
            #beta1=np.random.uniform(low=0.7, high=0.99) # m = b1m + (1 - b1)m
            #beta2=np.random.uniform(low=0.8, high=0.999) # v = b2 v + (1 - b2)v
            results['beta1']=float(beta1)
            results['beta2']=float(beta2)
        elif optimization_alg == 'RMSProp':
            decay = np.random.uniform(low=0.75,high=0.99)
            momentum = np.random.uniform(low=0.0,high=0.9)
            results['decay']=float(decay)
            results['momentum']=float(momentum)
        else:
            pass

        results['range_learning_rate'] = [low_const_learning_rate, high_const_learning_rate]
        #results['range_constant'] = [low_const, high_const]
    else:
        print '::::+++++====> Running MANUAL SETTING OF HYPER PARAMETERS'
        dims = [D]+units_list+[D_out]
        mu_init = 0.0
        mu = len(dims)*[mu_init]
        #std_init = 0.01
        std_init = 0.1
        std = len(dims)*[std_init]
        #std = [None, std_init,std_init,10]
        print 'std: ', std
        #low_const, high_const = 0.4, 1.0
        init_constant = 0.1
        #init_constant = 0.4177
        #init_constant = 2.0
        b_init = len(dims)*[init_constant]
        #b_init = [None, init_constant, 1.2]
        print '++> S/b_init ', b_init
        S_init = b_init
        #
        model = mdl_type
        #
        max_to_keep = 1

        phase_train = tf.placeholder(tf.bool, name='phase_train') if bn else  None

        report_error_freq = 25
        steps = 20000
        M = 8000 #batch-size
        print '++++> M (batch size) :', M

        starter_learning_rate = 0.01

        print '++> starter_learning_rate ', starter_learning_rate
        ## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decay_rate = 0.9
        decay_steps = 200
        staircase = True
        print '++> decay_rate ', decay_rate
        print '++> decay_steps ', decay_steps
        print '++> staircase ', staircase

        if optimization_alg=='Momentum':
            #use_nesterov=False
            momentum = 0.8
            results['momentum']=float(momentum)
        elif optimization_alg == 'Adadelta':
            rho = 0.95
            results['rho']=float(rho)
        elif optimization_alg == 'Adagrad':
            #only has learning rate
            pass
        elif optimization_alg == 'Adam':
            beta1=0.99 # m = b1m + (1 - b1)m
            beta2=nhigh=0.999 # v = b2 v + (1 - b2)v
            # w := w - m/(sqrt(v)+eps)
            results['beta1']=float(beta1)
            results['beta2']=float(beta2)
        elif optimization_alg =='RMSProp':
            decay = 0.9 #Discounting factor for the history/coming gradient
            #momentum = 0.85
            momentum = 0.9
            results['decay']=float(decay)
            results['momentum']=float(momentum)

    ##############################
    ##
    #X_reconstruct_pca, _, _ = mtf. get_reconstruction(X_train,k=units_list[0])
    #print '*************> PCA error: ', mtf.report_l2_loss(Y=X_train,Y_pred=X_reconstruct_pca)
    if task_name == 'task_MNIST_flat_auto_encoder':
        PCA_errors = {12:24.8254684915, 48:9.60052317906, 96:4.72118325768}
        if len(units_list) == 1:
            k = units_list[0]
        else:
            k = units_list[0] * len(units_list)
        if not k in PCA_errors.keys():
            print 'COMPUTING PCA... k = ', k
            X_reconstruct_pca, _, _ = mtf. get_reconstruction_pca(X_train,k=units_list[0])
            pca_error = mtf.report_l2_loss(Y=X_train,Y_pred=X_reconstruct_pca)
            PCA_errors[k] = pca_error
        else:
            pca_error = PCA_errors[k]
        print '*************> PCA error: ', pca_error
    else:
        pca_error = None
        rbf_error = None

    hbf1_error = None
    if model == 'hbf':
        #error, Y_pred, Kern, C, subsampled_data_points = report_RBF_error_from_data(X_train, dims, stddev)
        if len(units_list) > 1:
            k = units_list[0]*len(units_list)
            print 'RBF units = ', k
            nb_units = [None, k]
            rbf_error, _, _, _, _ = mtf.report_RBF_error_from_data(X_train, X_train, nb_units, S_init[1])
            print rbf_error
            hbf1={12:26.7595}
            if k in hbf1.keys():
                hbf1_error = hbf1[k]
        else:
            nb_units = dims
            rbf_error, _, _, _, _ = mtf.report_RBF_error_from_data(X_train, X_train, nb_units, S_init[1])

    S_init = b_init
    ##

    ## Make Model
    nb_layers = len(dims)-1
    nb_hidden_layers = nb_layers-1
    print( '-----> Running model: %s. (nb_hidden_layers = %d, nb_layers = %d)' % (model,nb_hidden_layers,nb_layers) )
    print( '-----> Units: %s)' % (dims) )
    if model == 'standard_nn':
        rbf_error = None
        #tensorboard_data_dump = '/tmp/standard_nn_logs'
        float_type = tf.float64
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D
        (inits_C,inits_W,inits_b) = mtf.get_initilizations_standard_NN(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train)
        with tf.name_scope("standardNN") as scope:
            mdl = mtf.build_standard_NN(x,dims,(inits_C,inits_W,inits_b),phase_train,trainable_bn)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
        inits_S = inits_b
    elif model == 'hbf':
        #tensorboard_data_dump = '/tmp/hbf_logs'
        float_type = tf.float64
        x = tf.placeholder(float_type, shape=[None, D], name='x-input') # M x D
        (inits_C,inits_W,inits_S,rbf_error) = mtf.get_initilizations_HBF(init_type=init_type,dims=dims,mu=mu,std=std,b_init=b_init,S_init=S_init, X_train=X_train, Y_train=Y_train, train_S_type=train_S_type)
        print inits_W
        with tf.name_scope("HBF") as scope:
            mdl = mtf.build_HBF2(x,dims,(inits_C,inits_W,inits_S),phase_train,trainable_bn,trainable_S)
            mdl = mtf.get_summation_layer(l=str(nb_layers),x=mdl,init=inits_C[0])
    elif model == 'binary_tree_4D_conv':
        print 'binary_tree_4D'
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
        #nb_filters = nb_filters
        mean, stddev = bn_tree_init_stats
        stddev = float( np.random.uniform(low=0.001, high=stddev) )
        print 'stddev', stddev
        x = tf.placeholder(float_type, shape=[None,1,D,1], name='x-input')
        with tf.name_scope("build_binary_model") as scope:
            mdl = mtf.build_binary_tree(x,filter_size,nb_filters,mean,stddev,stride_convd1=2,phase_train=phase_train,trainable_bn=trainable_bn)
        #
        dims = [D]+[nb_filters]+[D_out]
        results['nb_filters'] = nb_filters
    elif model == 'binary_tree_D8':
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
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
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
        print 'task_name: ', task_name
        print 're_train: ', re_train
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
            print ('>',msg)

    if use_tensorboard:
        if tf.gfile.Exists('/tmp/mdl_logs'):
          tf.gfile.DeleteRecursively('/tmp/mdl_logs')
        tf.gfile.MakeDirs('/tmp/mdl_logs')

    tf.add_check_numerics_ops()

    # Add ops to save and restore all the variables.
    if mdl_save:
        saver = tf.train.Saver(max_to_keep=max_to_keep)
    start_time = time.time()
    file_for_error = './ray_error_file.txt'
    #with open(path+errors_pretty, 'w+') as f_err_msgs:
    with open(file_for_error, 'w+') as f_err_msgs:
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
            for i in xrange(steps):
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
                    loss_msg+="\n"
                    mdl_info_msg+="\n"
                    errors_to_beat+="\n"

                    print 'S: ', inits_S
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
    results['tf_rand_seed'] = tf_rand_seed
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
    print '\a' #makes beep
    print '\a' #makes beep


# low_const, high_const = 0.4, 1.0
# #init_constant = np.random.uniform(low=low_const, high=high_const)
# #b_init = list(np.random.uniform(low=low_const, high=high_const,size=len(dims)))
# init_constant = 0.4177551
# #b_init = len(dims)*[init_constant]
# #[0.6374998052942504, 0.6374998052942504, 0.6374998052942504, 0.6374998052942504]
# b_init = [None, init_constant, np.random.uniform(low=1,high=2.5)]

if __name__ == '__main__':
    print 'in __main__'
    print 'start running main_nn.py'
    main()
    print 'end running main_nn.py'
