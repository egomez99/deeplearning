# Fernando Velasco
# Eduardo Romo
#-----------------------------------INSTRUCTIONS-----------------------------------
# Complete the code in the functions "sigmoid" and "predict" to implement a
# Feedfordward algorithm
#----------------------------------------------------------------------------------
import scipy.io as sio #Used to import data
import numpy as np #Used to manage arrays
import matplotlib.pyplot as pl #Used to display images

#-----------------------------------YOUR CODE START HERE-----------------------------------
def sigmoid(z):
    #calculate the sigmod value of a numpy array
    return 1 / (1 + np.exp(-z))

def predict (Theta1,Theta2,X):
    #Use the sigmoid function to calculate the activation in each neuron
    #Don'f forget to add the bias input into the input structure
    #Repeat for every example in X
    #Consider that the labels need to be increased by one
    #p = np.array(np.zeros(len(X)))
    input_layer=np.insert(X,0,values=np.ones(len(X)),axis=1)
    z2=np.dot(input_layer,np.transpose(Theta1))
    hidden_layer=np.insert(sigmoid(z2),0,values=np.ones(len(sigmoid(z2))),axis=1)
    z3=np.dot(hidden_layer,np.transpose(Theta2))
    ouput_layer=sigmoid(z3)

    return np.array(np.argmax(ouput_layer,axis=1)+1)

#-----------------------------------YOUR CODE END HERE-----------------------------------

#-----------------------------------Load Data-----------------------------------
inputData = sio.loadmat('ex3data1.mat')
weights = sio.loadmat('ex3weights.mat')

y = np.array(inputData["y"])
X = np.array(inputData["X"])
Theta1 = np.array(weights["Theta1"])
Theta2 = np.array(weights["Theta2"])

#-----------------------------------Visualize the Data-----------------------------------

indexes = np.array(range(5000))
np.random.shuffle(indexes)

for shape in range(16):#Show 16 random pictures and its corresponding value.
    index = indexes[shape]
    image = np.reshape(X[index],(-1,20),order='F')
    pl.subplot(4,4,shape+1)
    pl.xlabel("Value: "+str(y[index][0]))
    pl.imshow(image, cmap="Greys_r")

pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
pl.show()


#-----------------------------------Test Sigmoid Function-----------------------------------
sigmoidSmall = sigmoid(np.array([-99]))
sigmoidZero = sigmoid(np.array([0]))
sigmoidBig = sigmoid(np.array([99]))

print("The sigmoid ouputs shall be:")
print("Small: -99 ->",sigmoidSmall)
print("Zero: 0.5 ->",sigmoidZero)
print("Big: 99 ->",sigmoidBig)

input("Press any key to continue")

#-----------------------------------Predict Values-----------------------------------
print("Input Shape:",X.shape)
print("Labels Shape:",y.shape)
print("Theta1 weights shape:",Theta1.shape)
print("Theta2 weights shape:",Theta2.shape)

p = predict (Theta1,Theta2,X)

#Evaluate if the predicted values are equal to the input values.
compare = p==y.T
print('COMPARE: ', compare)
print("Precision of the network is:",(np.sum(compare))/(len(p)))
print("The Precision shall be close to 0.9752")

indexes = np.array(range(5000))

np.random.shuffle(indexes)

for shape in range(16):#Show 16 random pictures and its corresponding value.
    index = indexes[shape]
    image = np.reshape(X[index],(-1,20),order='F')
    pl.subplot(4,4,shape+1)
    pl.xlabel("P Value: "+str(predict(Theta1,Theta2,[X[index]])))
    pl.imshow(image, cmap="Greys_r")
pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
pl.show()
print("\nFinish")