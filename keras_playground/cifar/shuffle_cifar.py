from __future__ import print_function
from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils

from my_tf_pkg import shuffle_images_at_scales as siac

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
scales = [None, 0, 1]
images = siac.shuffle_at_scales(scales,X_train)

siac.plot_a_single_image(1,images)
