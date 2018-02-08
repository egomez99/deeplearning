import numpy as np
import random


learning_rate=0.2
training_data_NAND = [(np.array([0,0,1]), 1),
                      (np.array([0,1,1]), 1),
                      (np.array([1,0,1]), 1),
                      (np.array([1,1,1]), 0)]

def step_activation(x):
	return 0 if x < 0 else 1

def learning(training_set,weights, iterations):
	for i in range(iterations):
		x, expected=random.choice(training_set)
		result =np.dot(weights,x)
		error=expected-step_activation(result)
		print ("{} - {} : {}".format(result,error,weights))
		weights+=learning_rate * error * x

def testing(training_set,weights):
	for x,_ in training_set:
		result=np.dot(x,weights)
		print("{} : {}  -> {}".format(x[:2],result,step_activation(result)))

weights=np.random.rand(3)
iterations=5000
learning(training_data_NAND,weights,iterations)
testing(training_data_NAND,weights)

