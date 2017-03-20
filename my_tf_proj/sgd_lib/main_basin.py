import tensorflow as tf
import csv

import datetime

import my_tf_pkg as mtf
from my_tf_pkg import main_hp

import pdb

##

#def get_

##

def get_basin_loss_surface(arg):
    basins = arg.get_basins(arg)
    #
    loss = 1
    for i in range( len(basins) ):
        basin = basins[i]
        loss = loss - basin # loss := loss + - exp( - 1/2sig [x - mu]^2)
    return loss

def get_basin(W,init_std,init_mu,l):
    # make standard dev
    S = tf.get_variable(name='S'+l, initializer=init_std, trainable=False)
    beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float32),S), 2)
    # make basin
    mu = tf.get_variable(name='mu'+l, initializer=init_mu, trainable=False)
    basin = tf.exp( - beta * tf.matmul(W - mu, W - mu) )
    #basin = tf.exp( - beta * (W - mu)*(W - mu) )
    return basin

##

def main_basin(arg):
    '''
    '''
    # force to flushing to output as default
    print = arg.print_func
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
    #
    #### build graph
    graph = tf.Graph()
    with graph.as_default():
        ### get mdl
        w = tf.placeholder(arg.float_type, arg.get_x_shape(arg), name='x-input')
        ### get basin
        loss = get_basin_loss_surface(arg)
        accuracy = loss
        arg.loss = loss
        ### get optimizer variables
        opt = main_hp.get_optimizer(arg)
        train_step = opt
        #train_step = opt.minimize(loss, global_step=arg.global_step)
        # step for optimizer (useful for ckpts)
        step, nb_iterations = tf.Variable(0, name='step'), tf.Variable(arg.nb_steps, name='nb_iterations')
        batch_size = tf.Variable(arg.batch_size, name='batch_size')
        # save everything that was saved in the session
        saver = tf.train.Saver()
    #### run session
    with tf.Session(graph=graph) as sess:
        with open(arg.path_to_hp+arg.csv_errors_filename,mode='a') as errors_csv_f: # a option: Opens a file for appending. The file pointer is at the end of the file if the file exists. That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
            #writer = csv.Writer(errors_csv_f)
            writer = csv.DictWriter(errors_csv_f,['train_error', 'cv_error', 'test_error'])
            # if (there is a restore ckpt mdl restore it) else (create a structure to save ckpt files)
            if arg.restore:
                arg.restore = False # after the model has been restored, we continue normal until all hp's are finished
                saver.restore(sess=sess, save_path=arg.save_path_to_ckpt2restore) # e.g. saver.restore(sess=sess, save_path='./tmp/my-model')
                arg = main_hp.restore_hps(arg)
                print('restored model trained up to, STEP: ', step.eval())
                print('restored model, ACCURACY:', sess.run(fetches=accuracy, feed_dict={x: X_train, y_: Y_train, phase_train: False}))
            else: # NOT Restore
                # not restored, so its a virgin run from scratch for this hp
                main_hp.deleteContent(pfile=errors_csv_f) # since its a virgin run we
                writer.writeheader()
                #
                main_hp.save_hps(arg) # save current hyper params
                if arg.save_checkpoints or arg.save_last_mdl:
                    mtf.make_and_check_dir(path=arg.path_to_ckpt+arg.hp_folder_for_ckpt) # creates ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N
                sess.run(tf.global_variables_initializer())
            # tensorboard
            if arg.use_tensorboard:
                # set up tensor board
                if arg.use_tensorboard:
                    if tf.gfile.Exists(arg.tensorboard_data_dump_train):
                      tf.gfile.DeleteRecursively(arg.tensorboard_data_dump_train)
                    tf.gfile.MakeDirs(arg.tensorboard_data_dump_train)
                # set up summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(arg.tensorboard_data_dump_train,sess.graph)
                #test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
                loss = [merged, loss]
            # train
            start_iteration = step.eval() # last iteration trained is the first iteration for this model
            for i in range(start_iteration,nb_iterations.eval()):
                #batch_xs, batch_ys = mnist.train.next_batch(batch_size.eval())
                #batch_xs, batch_ys = main_hp.get_batch_feed(X_train, Y_train, batch_size.eval())
                #pdb.set_trace()
                sess.run(fetches=train_step)
                # check_point mdl
                if i % arg.report_error_freq == 0:
                    sess.run(step.assign(i))
                    #
                    train_summary, train_error = sess.run(fetches=loss)
                    #print( 'step %d, train error: %s | batch_size(step.eval(),arg.batch_size): %s,%s log_learning_rate: %s | mdl %s '%(i,train_error,batch_size.eval(),arg.batch_size,arg.log_learning_rate,arg.mdl) )
                    print( 'step %d, train error: %s | starter_learning_rate: %s | mdl %s '%(i,train_error,arg.starter_learning_rate, arg.mdl) )
                    # write files
                    writer.writerow({'train_error':train_error})
                    # save checkpoint
                    if arg.save_checkpoints:
                        saver.save(sess=sess,save_path=arg.path_to_ckpt+arg.hp_folder_for_ckpt+arg.prefix_ckpt)
                    # write tensorboard
                    if arg.use_tensorboard:
                        train_writer.add_summary(train_summary, i)
                # save last model
                if arg.save_last_mdl:
                    saver.save(sess=sess,save_path=arg.path_to_ckpt+arg.hp_folder_for_ckpt+arg.prefix_ckpt)
            # evaluate
            print('Final Test Acc/error: ', sess.run(fetches=accuracy))
