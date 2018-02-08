from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

indexes = np.array(range(5000))

np.random.shuffle(indexes)

for shape in range(16):
	index=indexes[shape]
	plt.subplot(4,4,shape+1)
	plt.imshow(X_train[index],cmap=plt.get_cmap('gray'))
	

plt.show()