import tensorflow as tf

'''
By default, TensorFlow maps nearly all of the GPU memory of all GPUs visible to the process.
This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory,
or to only grow the memory usage as is needed by the process.
TensorFlow provides two Config options on the Session to control this.

The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations:
it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed,
we extend the GPU memory region needed by the TensorFlow process.
Note that we do not release memory, since that can lead to even worse memory fragmentation.
To turn this option on, set the option in the ConfigProto by:
'''

# grows memory needed by gpu dynamically
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

'''
The second method is the per_process_gpu_memory_fraction option,
which determines the fraction of the overall amount of memory that each visible GPU should be allocated.
For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:
'''

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

#This is useful if you want to truly bound the amount of GPU memory available to the TensorFlow process.
