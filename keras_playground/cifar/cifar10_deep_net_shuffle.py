'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#params
nb_classes = 10

units_single_layer = 10000
data_augmentation = False

#units_single_layer = 10000
activation_func = 'relu'
activation_func = 'sigmoid'

nb_epoch = 25
batch_size = 64
#optimizer = 'adam'
optimizer = 'rmsprop'

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#create model
print('\n ---- Deep Layer Model ----')
print('units_single_layer: ', units_single_layer)
print('activation_func: ', activation_func)
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation(activation_func))

model.add(Convolution2D(32, 3, 3))
model.add(Activation(activation_func))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation(activation_func))

model.add(Convolution2D(64, 3, 3))
model.add(Activation(activation_func))

model.add(Flatten())
model.add(Dense(512))

model.add(Activation(activation_func))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Not using data augmentation.')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
