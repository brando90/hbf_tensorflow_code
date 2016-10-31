import tensorflow as tf

#might need to add cuda and cdnn (in polestar)
#module add cudnn/5
#module add cuda/7.5

#If you would like a particular operation to run on a device of your choice instead of
#what's automatically selected for you, you can use with tf.device to create a device
#context such that all the operations within that context will have the same device assignment.

# Creates a graph
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
#log_device_placement basically "logs"/prints which devices stuff is sent to
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print( sess.run(c) )
