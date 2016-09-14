import numpy as np

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def hello_world():
    print( "Hello World!" )

## builders for Networks

# def build_HBF(x, dims, inits, phase_train=None, trainable=True):
#     (_,inits_W,inits_S) = inits
#     layer = x
#     nb_hidden_layers = len(dims)-1
#     for l in range(1,nb_hidden_layers): # from 1 to L-1
#         layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
#     return layer

def build_standard_NN(arg, x, dims, inits, phase_train=None, trainable_bn=True):
    (_,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(arg, l=str(l), x=layer, init=(inits_W[l],inits_b[l]), dims=(dims[l-1], dims[l]), phase_train=phase_train, trainable_bn=trainable_bn)
    return layer

## build layers blocks NN

def build_HBF2(x, dims, inits, phase_train=None, trainable_bn=True,trainable_S=True):
    (_,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        #print nb_hidden_layers
        #print len(inits_W)
        layer = get_HBF_layer2(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train, trainable_bn=trainable_bn,trainable_S=trainable_S)
        #layer = get_HBF_layer3(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train, trainable_bn=trainable_bn,trainable_S=trainable_S)
    return layer

def get_HBF_layer2(l, x, dims, init, phase_train=None, layer_name='HBFLayer', trainable_bn=True, trainable_S=True):
    (init_W,init_S) = init
    with tf.name_scope(layer_name+l):
        with tf.name_scope('templates'+l):
            #W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
            W = get_W(init_W, l, dims)
        with tf.name_scope('rbf_stddev'+l):
            print( '--> init_S: ', init_S)
            print('--> trainable_S: ', trainable_S)
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=trainable_S)
            beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
        with tf.name_scope('Z'+l):
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) # (1 x D^(l)) = sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
            Z = beta * ( Delta_tilde ) # (M x D^(l))
        # if phase_train is not None:
        #     Z = add_batch_norm_layer(l, Z , phase_train, trainable_bn=trainable_bn)
        #     Z = tf.abs(Z)
            #A = tf.exp(-Z) # (M x D^(l))
        with tf.name_scope('A'+l):
            A = tf.exp(Z) # (M x D^(l))
        # if phase_train is not None:
        #     A = add_batch_norm_layer(l, A , phase_train, trainable_bn=trainable_bn)
            #z = add_batch_norm_layer(l, z, phase_train, trainable_bn=trainable_bn)
    var_prefix = 'vars_'+layer_name+l
    put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
    put_summaries(var=S,prefix_name=var_prefix+S.name,suffix_text=S.name)
    act_stats = 'acts_'+layer_name+l
    put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
    put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
    put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
    put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

def get_HBF_layer3(l, x, dims, init, phase_train=None, layer_name='HBFLayer', trainable_bn=True, trainable_S=True):
    (init_W,init_S) = init
    with tf.name_scope(layer_name+l):
        with tf.name_scope('templates'+l):
            #W = tf.get_variable(name='W'+l, dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
            W = get_W(init_W, l, dims)
        with tf.name_scope('rbf_stddev'+l):
            print( '--> init_S: ', init_S)
            print( '--> trainable_S: ', trainable_S)
            S = tf.get_variable(name='S'+l, dtype=tf.float64, initializer=init_S, regularizer=None, trainable=trainable_S)
            beta = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
        with tf.name_scope('Z'+l):
            WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) # (1 x D^(l)) = sum( (D^(l-1) x D^(l)), 0 )
            XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
            # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
            Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX) # (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))

            #Delta_tilde = tf.Print(Delta_tilde,[Delta_tilde], message="my Delta_tilde-values:",first_n=10)

            Z = beta * ( Delta_tilde ) # (M x D^(l))
            #Z = tf.Print(Z,[Z], message="Z:",first_n=10)
        if phase_train is not None:
            Z = add_batch_norm_layer(l, Z , phase_train, trainable_bn=trainable_bn)
            #Z = tf.Print(Z,[Z], message="Z:",first_n=10)
        with tf.name_scope('A'+l):
            Y = tf.square(Z)
            #Y = tf.Print(Y,[Y], message="Z:",first_n=10)
            #
            init_a = tf.constant(2.0,dtype=tf.float64)
            a = tf.get_variable(name='a'+l, dtype=tf.float64, initializer=init_a, regularizer=None, trainable=True)
            precision = tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),a), 2)
            precision = 1.0
            #
            A = tf.exp(-precision*Y) # (M x D^(l))
    var_prefix = 'vars_'+layer_name+l
    put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
    put_summaries(var=S,prefix_name=var_prefix+S.name,suffix_text=S.name)
    act_stats = 'acts_'+layer_name+l
    put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
    put_summaries(Y,prefix_name=act_stats+'Y'+l,suffix_text='Y'+l)
    put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
    put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
    put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

def put_summaries(var, prefix_name, suffix_text = ''):
    """Attach a lot of summaries to a Tensor."""
    prefix_title = prefix_name+'/'
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary(prefix_title+'mean'+suffix_text, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary(prefix_title+'stddev'+suffix_text, stddev)
        tf.scalar_summary(prefix_title+'max'+suffix_text, tf.reduce_max(var))
        tf.scalar_summary(prefix_title+'min'+suffix_text, tf.reduce_min(var))
        tf.histogram_summary(prefix_name, var)

# def put_summaries_absolute_val(var, prefix_name, suffix_text = ''):
#     """Attach a lot of summaries to a Tensor to check is absolute value"""
#     prefix_title = prefix_name+'/'
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary(prefix_title+'mean'+suffix_text, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#         tf.scalar_summary(prefix_title+'stddev'+suffix_text, stddev)
#         tf.scalar_summary(prefix_title+'max'+suffix_text, tf.reduce_max(var))
#         tf.scalar_summary(prefix_title+'min'+suffix_text, tf.reduce_min(var))
#         tf.histogram_summary(prefix_name, var)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def get_NN_layer(arg, l, x, dims, init, phase_train=None, scope="NNLayer", trainable_bn=True):
    (init_W,init_b) = init
    with tf.name_scope(scope+l):
        #print( 'init_W ', init_W )
        W = get_W(init_W, l, dims)
        b = tf.get_variable(name='b'+l, dtype=tf.float64, initializer=init_b, regularizer=None, trainable=True)
        with tf.name_scope('Z'+l):
            Z = tf.matmul(x,W) + b
            if phase_train is not None:
                #z = standard_batch_norm(l, z, 1, phase_train)
                Z = add_batch_norm_layer(l, Z, phase_train, trainable_bn=trainable_bn)
        with tf.name_scope('A'+l):
            #A = tf.nn.relu(Z) # (M x D1) = (M x D) * (D x D1)
            A = arg.act(Z)
    with tf.name_scope('sumarries'+l):
        #W = tf.histogram_summary('W'+l, W)
        #b = tf.histogram_summary('b'+l, b)
        layer_name = scope
        var_prefix = 'vars_'+layer_name+l
        put_summaries(var=W,prefix_name=var_prefix+W.name,suffix_text=W.name)
        put_summaries(var=b,prefix_name=var_prefix+b.name,suffix_text=b.name)
        act_stats = 'acts_'+layer_name+l
        put_summaries(Z,prefix_name=act_stats+'Z'+l,suffix_text='Z'+l)
        put_summaries(A,prefix_name=act_stats+'A'+l,suffix_text='A'+l)
        #put_summaries(Delta_tilde,prefix_name=act_stats+'Delta_tilde'+l,suffix_text='Delta_tilde'+l)
        #put_summaries(beta,prefix_name=act_stats+'beta'+l,suffix_text='beta'+l)
    return A

## 4D BT

def build_binary_tree_4D_hidden_layer(x,arg,phase_train=None):
    ## 1st hidden conv layer
    Z1 = get_binary_branch(x,arg,l=0,name='Conv_Layer') # N x D_conv_flat = N x (filter_size*nb_filters)
    if phase_train is not None:
        Z1 = add_batch_norm_layer(l='BN1',x=Z1,phase_train=phase_train,trainable_bn=arg.trainable_bn)
    A1 = arg.act( Z1 ) # N x D_conv_flat = N x (filter_size*nb_filters)
    ## 2nd hidden layer
    b2 = tf.Variable(tf.constant(0.1, shape=[arg.nb_final_hidden_units]))
    W2 = get_W_BT4D(arg,l=1,name='hidden_layer',dtype=tf.float32)
    Z2 = tf.matmul(A1,W2) + b2
    if phase_train is not None:
        Z2 = add_batch_norm_layer(l='BN2',x=Z2,phase_train=phase_train,trainable_bn=trainable_bn)
    A2 = arg.act( Z2 )
    # 3rd fully connected layer
    init_C = tf.truncated_normal(shape=[arg.nb_final_hidden_units,1], mean=arg.mean[2], stddev=arg.stddev[2], dtype=tf.float32, seed=None, name=None)
    C = tf.get_variable(name='W'+'Out_Layer',dtype=tf.float32,initializer=init_C,regularizer=None,trainable=True)
    f = tf.matmul(A2,C)
    return f

def get_binary_branch(x,arg,l,name=None):
    '''
        stride_convd1 # controls the stride for 1D convolution
    '''
    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    W_filters = get_W_conv2d(arg,l,name) # [1,arg.filter_size,1,arg.nb_filters]
    b = tf.Variable( tf.constant(0.1, shape=[arg.nb_filters]) )
    # 1D conv
    conv = tf.nn.conv2d(input=x, filter=W_filters, strides=[1, 1, arg.stride_convd1, 1], padding="SAME", name="conv")
    flat_conv = tf.reshape(conv + b, [-1,arg.filter_size*arg.nb_filters])
    return flat_conv

# def build_binary_tree(x,filter_size,nb_filters,mean,stddev,stride_convd1=2,phase_train=None,trainable_bn=True):
#     ## conv layer
#     l = 'Conv_Layer'
#     flat_conv = get_binary_branch(l,x,filter_size,nb_filters,mean=mean,stddev=stddev,stride_convd1=stride_convd1) # N x D_conv_flat = N x (filter_size*nb_filters)
#
#     if phase_train is not None:
#         l = 'BN'
#         flat_conv = add_batch_norm_layer(l, flat_conv, phase_train, trainable_bn=trainable_bn)
#
#     ## fully connected layer
#     init_W = tf.truncated_normal(shape=[filter_size*nb_filters,1], mean=mean, stddev=stddev, dtype=tf.float32, seed=None, name=None)
#     print( '-->-->-->-->-->-->-->-->-->-->-->init_C: ', init_W)
#     l = 'Out_Layer'
#     C = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
#     mdl = tf.matmul(flat_conv,C)
#     return mdl

##

def build_binary_tree_8D(x,nb_filters1,nb_filters2,mean1=0.0,stddev1=0.1,mean2=0.0,stddev2=0.01,mean3=0.0,stddev3=0.1,stride_conv1=2):
    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    filter_size1 = 2
    # W
    l='conv1'
    init_W = tf.truncated_normal(shape=[1,filter_size1,1,nb_filters1], mean=mean1, stddev=stddev1, dtype=tf.float32, seed=None, name=None)
    W_filters = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
    #bias
    b1 = tf.Variable( tf.constant(0.1, shape=[nb_filters1]) )
    # BT1
    x1 = tf.slice(x, begin=[0,0,0,0],size=[-1,-1,4,-1], name=None)
    x2 = tf.slice(x, begin=[0,0,4,0],size=[-1,-1,4,-1], name=None)
    Y_11 = get_binary_subtree(l='Y11',x=x1,W_filters=W_filters,b=b1,nb_filters=nb_filters1,stride_convd1=stride_conv1) # M, 2 x nb_filters1
    Y_12 = get_binary_subtree(l='Y12',x=x2,W_filters=W_filters,b=b1,nb_filters=nb_filters1,stride_convd1=stride_conv1) # M, 2 x nb_filters1
    Y_1 = tf.concat(1, [Y_11, Y_12]) # M, 4 x nb_filters
    Y_1 = tf.reshape(Y_1, [-1,1,(2*filter_size1)*nb_filters1,1]) # [N, 1, D1, 1] = [N, 1, 2d1, 1]
    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    d1 = filter_size1*nb_filters1
    stride_conv2 = d1
    filter_size2 = d1
    # W_tilde
    l='conv2'
    init_W_tilde = tf.truncated_normal(shape=[1,filter_size2,1,nb_filters2], mean=mean2, stddev=stddev2, dtype=tf.float32, seed=None, name=None)
    W_filters_tilde = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W_tilde, regularizer=None, trainable=True)
    # biases
    b2 = tf.Variable( tf.constant(0.1, shape=[nb_filters2]) )
    # BT2
    Y_21 = get_binary_subtree(l='Y21',x=Y_1,W_filters=W_filters_tilde,b=b2,nb_filters=nb_filters2,stride_convd1=stride_conv2) # M, 2 x nb_filters2
    d_out = 2*nb_filters2
    Y_2 = tf.reshape(Y_21, [-1,d_out])
    # Out
    l = 'Out_Layer'
    init_C = tf.truncated_normal(shape=[d_out,1], mean=mean3, stddev=stddev3, dtype=tf.float32, seed=None, name=None)
    C = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_C, regularizer=None, trainable=True)
    mdl = tf.matmul(Y_2,C)
    return mdl

def get_binary_subtree(l,x,W_filters,b,nb_filters,stride_convd1=2):
    '''
        x = (M, 1, D, 1)
        b = [nb_filters]
        W_filters = (1,filter_size,1,nb_filters)  == "[filter_height, filter_width, in_channels, out_channels]"
        filter shape is "[filter_height, filter_width, in_channels, out_channels]"
        stride_convd1 # controls the stride for 1D convolution
    '''
    # 1D conv
    # M, 1, 2, nb_filters
    conv = tf.nn.conv2d(input=x, filter=W_filters, strides=[1, 1, stride_convd1, 1], padding="VALID", name="conv")
    # get activations
    A = tf.nn.relu( conv + b ) # M, 1, 2, nb_filters
    A_flat = tf.reshape(A, [-1,2*nb_filters]) # M , filter_size*nb_filters
    return A_flat

#

def build_binary_tree_16D(x,nb_filters1,nb_filters2,mean,stddev,stride_convd1=2):
    # # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    # filter_size1 = 2
    # init_W = tf.truncated_normal(shape=[1,filter_size,1,nb_filters], mean=mean, stddev=stddev, dtype=tf.float32, seed=None, name=None)
    # W_filters = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
    # #bias
    # b = tf.Variable( tf.constant(0.1, shape=[nb_filters]) )
    # # left BT
    # Y_11 = get_binary_subtree(l='Y11',x=x[0:4],W_filters=W_filters,b,stride_convd1=2) # M, 2 x nb_filters
    # Y_12 = get_binary_subtree(l='Y12',x=x[4:8],W_filters=W_filters,b,stride_convd1=2) # M, 2 x nb_filters
    # # right BT
    # Y_13 = get_binary_subtree(l='Y13',x=x[8:12],W_filters=W_filters,b,stride_convd1=2) # M, 2 x nb_filters
    # Y_14 = get_binary_subtree(l='Y14',x=x[12:16],W_filters=W_filters,b,stride_convd1=2) # M, 2 x nb_filters
    # #
    # Y_1 = tf.concat(1, [Y_11, Y_12]) # M, 4 x nb_filters
    # Y_2 = tf.concat(1, [Y_13, Y_14]) # M, 4 x nb_filters
    # #
    # stride_convd1 = 4*nb_filters1 # unhard code it
    #
    # init_W_tilde = tf.truncated_normal(shape=[1,4*nb_filters1,1,nb_filters2], mean=mean, stddev=stddev, dtype=tf.float32, seed=None, name=None)
    # W_filters_tilde = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W_tilde, regularizer=None, trainable=True)
    # #
    # l = 'Out_Layer'
    # init_W = tf.truncated_normal(shape=[filter_size*nb_filters,1], mean=mean, stddev=stddev, dtype=tf.float32, seed=None, name=None)
    # C = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
    # mdl = tf.matmul(flat_conv,C)
    return mdl

##

def get_W_conv2d(arg,l,name,dtype=tf.float32):
    #init_W = tf.truncated_normal(shape=[1,filter_size,1,nb_filters], mean=mean, stddev=stddev, dtype=tf.float32, seed=None, name=None)
    #W_filters = tf.get_variable(name='W'+name, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
    if arg.init_type == 'xavier':
        init_W = tf.contrib.layers.xavier_initializer_conv2d(uniform=True,dtype=dtype)
        W = tf.get_variable(name='W'+name,dtype=dtype,initializer=init_W,regularizer=None,trainable=True,shape=[1,arg.filter_size,1,arg.nb_filters])
    else:
        init_W = tf.truncated_normal(shape=[1,arg.filter_size,1,arg.nb_filters], mean=arg.mean[l], stddev=arg.stddev[l], dtype=dtype)
        W = tf.get_variable(name='W'+name, dtype=dtype, initializer=init_W, regularizer=None, trainable=True)
    return W

def get_W_BT4D(arg,l,name,dtype=tf.float32):
    # init_W = tf.truncated_normal(shape=[filter_size*nb_filters,nb_final_hidden], mean=mean[1], stddev=stddev[1], dtype=tf.float32, seed=None, name=None)
    # W = tf.get_variable(name='W'+l, dtype=tf.float32, initializer=init_W, regularizer=None, trainable=True)
    dim_input,dim_out = arg.filter_size*arg.nb_filters, arg.nb_final_hidden_units
    if arg.init_type == 'xavier':
        init_W =tf.contrib.layers.xavier_initializer(dtype=dtype)
        W = tf.get_variable(name='W'+name,dtype=dtype,initializer=init_W,regularizer=None,trainable=True,shape=[dim_input,dim_out])
    else:
        init_W = tf.truncated_normal(shape=[dim_input,dim_out], mean=arg.mean[l], stddev=arg.stddev[l], dtype=dtype)
        W = tf.get_variable(name='W'+name, dtype=dtype, initializer=init_W, regularizer=None, trainable=True)
    return W

def get_W(init_W,l,dims,dtype=tf.float64):
    if isinstance(init_W, tf.python.framework.ops.Tensor):
        W = tf.get_variable(name='W'+l, dtype=dtype, initializer=init_W, regularizer=None, trainable=True)
    else:
        (dim_input,dim_out) = dims
        W = tf.get_variable(name='W'+l, dtype=dtype, initializer=init_W, regularizer=None, trainable=True, shape=[dim_input,dim_out])
    return W

def add_batch_norm_layer(l, x, phase_train, n_out=1, scope='BN', trainable_bn=True):
    print( 'add_batch_norm_layer')
    #phase_train = True
    #bn_layer = standard_batch_norm(l, x, n_out, phase_train, scope='BN')
    #bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l)
    bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l,trainable=trainable_bn)
    #bn_layer = batch_norm_layer(x,phase_train,scope_bn=scope+l,trainable=False)
    return bn_layer

def batch_norm_layer(x,phase_train,scope_bn,trainable=True):
    center = True
    scale = True
    print( '======> official BN')
    print( '--> trainable_bn: ', trainable)
    bn_train = batch_norm(x, decay=0.999, center=center, scale=scale,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=trainable,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=center, scale=scale,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=trainable,
    scope=scope_bn)
    z = tf.cond(phase_train, lambda: bn_train, lambda: bn_inference)
    return z

##

def BatchNorm_my_github_ver(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=False, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   updates_collections=None, center=False, scope=scope, reuse = True))

def BatchNorm_GitHub_Ver(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                lambda: batch_norm(inputT, is_training=True,
                                   center=False, updates_collections=None, scope=scope),
                lambda: batch_norm(inputT, is_training=False,
                                   updates_collections=None, center=False, scope=scope, reuse = True))

##

def build_summed_NN(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_b) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        layer = get_NN_layer(str(l),layer,dims,(inits_W[l],inits_b[l]), phase_train=None)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def build_summed_HBF(x, dims, inits, phase_train=None):
    (inits_C,inits_W,inits_S) = inits
    layer = x
    nb_hidden_layers = len(dims)-1
    for l in range(1,nb_hidden_layers): # from 1 to L-1
        layer = get_HBF_layer(l=str(l),x=layer,init=(inits_W[l],inits_S[l]),dims=(dims[l-1],dims[l]),phase_train=phase_train)
        layer = get_summation_layer(str(l),layer,inits_C[l])
    return layer

def get_summation_layer(l, x, init, layer_name="SumLayer"):
    with tf.name_scope(layer_name+l):
        #print init
        C = tf.get_variable(name='C', dtype=tf.float64, initializer=init, regularizer=None, trainable=True)
        layer = tf.matmul(x, C)
    var_prefix = 'vars_'+layer_name+l
    put_summaries(C, prefix_name=var_prefix+'C', suffix_text = 'C')
    return layer

def standard_batch_norm(l, x, n_out, phase_train, scope='BN'):
    """
    Batch normalization on feedforward maps.
    Args:
        x:           Vector
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope+l):
        #beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float64 ), name='beta', trainable=True, dtype=tf.float64 )
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out],dtype=tf.float64 ), name='gamma', trainable=True, dtype=tf.float64 )
        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float64)
        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float64)
        beta = tf.get_variable(name='beta'+l, dtype=tf.float64, initializer=init_beta, regularizer=None, trainable=True)
        gamma = tf.get_variable(name='gamma'+l, dtype=tf.float64, initializer=init_gamma, regularizer=None, trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

# def nn_layer(x, input_dim, output_dim, layer_name, act=tf.nn.relu, trainable=True):
#     """Reusable code for making a simple neural net layer.
#     It does a matrix multiply, bias add, and then uses relu to nonlinearize.
#     It also sets up name scoping so that the resultant graph is easy to read,
#     and adds a number of summary ops.
#     """
#     # Adding a name scope ensures logical grouping of the layers in the graph.
#     with tf.name_scope(layer_name):
#         # This Variable will hold the state of the weights for the layer
#         with tf.name_scope('weights'):
#             W = weight_variable([input_dim, output_dim])
#             variable_summaries(W, layer_name + '/weights')
#         with tf.name_scope('biases'):
#             b = bias_variable([output_dim])
#             variable_summaries(b, layer_name + '/biases')
#         with tf.name_scope('Wx_plus_b'):
#             Z = tf.matmul(x, W) + b
#             tf.histogram_summary(layer_name + '/pre_activations', Z)
#         A = act(Z, 'activation')
#         tf.histogram_summary(layer_name + '/activations', A)
#         return A
