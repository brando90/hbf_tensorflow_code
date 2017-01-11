import tensorflow as tf

def hello_world():
    print('hellow world')

def test_matrix_multipl():
    # Create a node Constant op that produces a 1x2 matrix.
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])

    # Create a node Matmul op
    product = tf.matmul(matrix1, matrix2)

    ## The Session closes automatically at the end of the with block.
    with tf.Session() as sess:
      result = sess.run(product)
      print('print product of matrices in TensorFlow')
      print(result)
      #print(type(result))
