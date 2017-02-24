# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import cifar10
from keras.utils import np_utils

import matplotlib.pyplot as plt

import numpy
import pickle

history = pickle.load( open( "history.p", "rb" ) )

# list all data in history
print(history.keys())
# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
