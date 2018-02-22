from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation,Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import os

EPOCHS=10
batch_size=128

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

indexes = np.array(range(5000))

np.random.shuffle(indexes)

weights=np.random.rand(784)


for shape in range(16):
	index=indexes[shape]
	plt.subplot(4,4,shape+1)
	plt.tight_layout()
	plt.imshow(X_train[index],cmap=plt.get_cmap('gray'))
	plt.title("Valor: {}".format(Y_train[index]))
	plt.xticks([])
	plt.yticks([])
#plt.show()

X_train=X_train.reshape(len(X_train),784)
X_train = X_train.astype('float32')
X_train/=255

X_test=X_test.reshape(len(X_test),784)
X_test = X_test.astype('float32')
X_test/=255

n_classes = 10
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_test = np_utils.to_categorical(Y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


model =Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(X_test, Y_test))

# saving the model
model_name = 'keras_model_1000.h5'
model_weights='mnist_weight_1000.h5'
model.save(model_name)
model.save_weights(model_weights)
print('Saved trained model at %s ' % model_name)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score)