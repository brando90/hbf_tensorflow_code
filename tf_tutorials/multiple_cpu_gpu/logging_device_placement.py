import tensorflow as tf

#might need to add cuda and cdnn (in polestar)
#module add cudnn/5
#module add cuda/7.5

#To find out which devices your operations and tensors are assigned to, create the session with log_device_placement configuration option set to True.

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
a_b = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print( sess.run(a_b) )
