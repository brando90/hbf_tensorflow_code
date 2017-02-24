# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import cifar10
from keras.utils import np_utils

import numpy
import pickle

print('start visualization example')

#params
nb_classes = 10

data_augmentation = False

units_single_layer = 10
actication_func = 'relu'
actication_func = 'sigmoid'

nb_epoch = 3
batch_size = 64
#optimizer = 'adam'
optimizer = 'rmsprop'

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape((X_train.shape[0],32*32*3))
X_test = X_test.reshape((X_test.shape[0],32*32*3))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# create model
print('\n ---- Singled Layer Model ----')
print('units_single_layer: ', units_single_layer)
print('actication_func: ', actication_func)
model = Sequential()

model.add(Dense(units_single_layer, input_shape=(32*32*3,)))
model.add(Activation(actication_func))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile model
print('\n ---- Optimizer ----')
print('optimizer: ', optimizer)
print('batch_size: ', optimizer)

print('')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# Fit the model
#history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, verbose=0)
history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

pickle.dump( history.history, open( "history.p", "wb" ) )
