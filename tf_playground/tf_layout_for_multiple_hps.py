def train_multiple_modles_in_one_script_with_gpu(arg):
    '''
    trains multiple NN models in one session using GPUs correctly.

    arg = some obj/struct with the params for trianing each of the models.
    '''
    #### try mutliple models
    for mdl_id in range(100):
        #### define/create graph
        graph = tf.Graph()
        with graph.as_default():
            ### get mdl
            x = tf.placeholder(float_type, get_x_shape(arg), name='x-input')
            y_ = tf.placeholder(float_type, get_y_shape(arg))
            y = get_mdl(arg,x)
            ### get loss and accuracy
            loss, accuracy = get_accuracy_loss(arg,x,y,y_)
            ### get optimizer variables
            opt = get_optimizer(arg)
            train_step = opt.minimize(loss, global_step=global_step)
        #### run session
        with tf.Session(graph=graph) as sess:
            # train
            for i in range(nb_iterations):
                batch_xs, batch_ys = get_batch_feed(X_train, Y_train, batch_size)
                sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
                # check_point mdl
                if i % report_error_freq == 0:
                    sess.run(step.assign(i))
                    #
                    train_error = sess.run(fetches=loss, feed_dict={x: X_train, y_: Y_train})
                    test_error = sess.run(fetches=loss, feed_dict={x: X_test, y_: Y_test})
                    print( 'step %d, train error: %s test_error %s'%(i,train_error,test_error) )
