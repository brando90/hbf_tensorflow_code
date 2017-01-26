from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# or You can also simply add layers via the .add() method:

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
