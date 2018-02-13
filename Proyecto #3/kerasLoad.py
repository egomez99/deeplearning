from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation,Dropout
import matplotlib.pyplot as plt
import numpy as np
import os


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

X_test=X_test.reshape(len(X_test),784)
X_test = X_test.astype('float32')
X_test/=255

# load the model and create predictions on the test set
model = load_model("keras_model_1000.h5")

model.load_weights("mnist_weight_1000.h5")

loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

