from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation,Dropout
import matplotlib.pyplot as plt
import numpy as np

def step_activation(x):
	return 0 if x < 0 else 1


def training(x_train,y_train,weights,iterations,learning_rate):
	for it in range(iterations):
		for shape in range(len(x_train)):
				image =x_train[shape]
				#print("shape w:{} , shape x:{}".format(weights.shape,image.shape))
				result=np.dot(weights[shape],image)
				print("w:{} \n\n image:{} \n\n result:{}".format(weights[shape],image,result))
				print(y_train[shape])
				error=y_train[shape]-step_activation(result)
				weights[shape]+=learning_rate*error*image
				print("result:{}, error:{}, w:{}".format(result,error,weights[shape]))

def training2(x_train,y_train,weights,iterations,learning_rate):
	for it in range(iterations):
		result=np.dot(weights,x_train.T)
		error=y_train[shape]-step_activation(result)
		print(result)

ITERATION_STEPS=1000

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

indexes = np.array(range(5000))

np.random.shuffle(indexes)

weights=np.random.rand(784)

print("shape w:{} shape x:{}".format(weights.shape,X_train.shape))

#X_train=np.reshape(X_train,(len(X_train),784))
#print("shape w:{} shape x:{}".format(weights.shape,X_train.shape))
# print("Training Len:",len(X_train))
# print("Weights Array: ",weights)
# print("Weights Len: ",len(weights))
for shape in range(16):
	index=indexes[shape]
	plt.subplot(4,4,shape+1)
	plt.tight_layout()
	plt.imshow(X_train[index],cmap=plt.get_cmap('gray'))
	plt.title("Valor: {}".format(Y_train[index]))
	plt.xticks([])
	plt.yticks([])
plt.show()

X_train=X_train.reshape(len(X_train),784)
X_train = X_train.astype('float32')
X_train/=255


model =Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer='sgd')

print("weights:{}".format(model.trainable_weights))


#training2(X_train,Y_train,weights,ITERATION_STEPS,.02)
#training(X_train,Y_train,weights,ITERATION_STEPS,.02)