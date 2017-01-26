# We should start by creating a TensorFlow session and registering it with Keras.
# This means that Keras will use the session we registered to initialize all variables that it creates internally.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.metrics import categorical_accuracy as accuracy
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.layers import Dense

mnist_data = input_data.read_data_sets('tmp_MNIST_data', one_hot=True)

# Now let's get started with our MNIST model. We can start building a classifier exactly as you would do in TensorFlow:
# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

# We define the placeholder for the labels, and the loss function we will use:
labels = tf.placeholder(tf.float32, shape=(None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with tf.Session() as sess:
    K.set_session(sess)

    # train
    sess.run(tf.global_variables_initializer())
    for i in range(1001):
        batch = mnist_data.train.next_batch(100)
        train_step.run(feed_dict={img: batch[0], labels: batch[1]})

    # evaluate
    acc_value = accuracy(labels, preds)
    print( acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels}) )
