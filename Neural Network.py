import numpy as np

import time
import os
import math
import time
import random

####################################################################################################
######### Importing training data and making it a numpy array for neural network to train on #######
####################################################################################################
os.system('clear')
j = 0
fh = open("mnist.txt")                                          # file with training data
train_x = []                                                    # temporary array
x = np.zeros((784, 3500))
train_y = []                                                    # temporary output array
xd = []
y = []
file = fh.read()
li = file.split("|")                                            # split input file into separate data points
for i in range(len(li)):
  li[i] = li[i].split(" ")                                      # split each data point into individual pixels
  j = 0
  while j < len(li[i]):
    if len(li[i][j]) < 4 and len(li[i][j]) > 0:                 # all numbers are 3 or less digits
      if ord(li[i][j][0]) < 48 or ord(li[i][j][0]) > 57:        # if not a number, remove from the list
        li[i].pop(j)
        j -= 1
      else:
        if "\n" in li[i][j]:
          li[i][j] = li[i][j].replace("\n", "")                 # it it is a number, remove any newline characters
        li[i][j] = int(li[i][j])
    else:                                                       # if more than 3 digits, remove from list
      li[i].pop(j)
      j -= 1
    j += 1
  if len(li[i]) == 10:                                          # put expected output data into y list
    ma = li[i][0]
    index = 0
    for k in range(1, len(li[i])):
      if li[i][k] > ma:
        ma = li[i][k]
        index = k
    xd.append(index)
    y.append(xd)
    xd = []
  elif len(li[i]) == 784:                                       # put input data into x list
    train_x.append(li[i])

train_x = np.array(train_x)

x = train_x.T

####################################################################################################
########## Importing testing data and making it a numpy array for neural network to test on ########
####################################################################################################
j = 0
fh = open("mnist2.txt")
train_x = []
x2 = np.zeros((784, 2030))
train_y = []
xd = []
y2 = []
file = fh.read()
li = file.split("|")
for i in range(len(li)):
  li[i] = li[i].split(" ")
  j = 0
  while j < len(li[i]):
    if len(li[i][j]) < 4 and len(li[i][j]) > 0:
      if ord(li[i][j][0]) < 48 or ord(li[i][j][0]) > 57:
        li[i].pop(j)
        j -= 1
      else:
        if "\n" in li[i][j]:
          li[i][j] = li[i][j].replace("\n", "")
        li[i][j] = int(li[i][j])
    else:
      li[i].pop(j)
      j -= 1
    j += 1
  if len(li[i]) == 10:
    ma = li[i][0]
    index = 0
    for k in range(1, len(li[i])):
      if li[i][k] > ma:
        ma = li[i][k]
        index = k
    xd.append(index)
    y2.append(xd)
    xd = []
  elif len(li[i]) == 784:
    train_x.append(li[i])

train_x = np.array(train_x)

x2 = train_x.T

####################################################################################################
#################### Convert values from larger numbers to numbers between 0 & 1 ###################
####################################################################################################
def divide(x):
  return x/255
div = np.vectorize(divide)                                          # convert function to a function that can be called on a numpy array

def divide2(x):
  return x/100
div2 = np.vectorize(divide2)                                        # convert function to a function that can be called on a numpy array

####################################################################################################
############## Print handwritten number by using asterisks to represent handwriting ################
####################################################################################################
def showNum(x, i):
  pri = []
  pr = []
  for k in range(28):
    for l in range(28):
      if x[:,i][28*k+l] > 0:                                        # if array's pixel value is greater than 0, draw an asterisk
        pr.append("*")
        pr.append(" ")
      else:
        pr.append(" ")                                              # else, leave it blank
        pr.append(" ")
    pri.append(pr)
    pr = []

  output = ""
  for m in range(len(pri)):
    for n in range(len(pri[m])):
      output += pri[m][n]
      output += ""
    print(output)
    output = ""

####################################################################################################
#################### Initializing weights, biases, and testing/training data #######################
#################################################################################################### 
weights1 = np.zeros((32,784))
weights2 = np.zeros((10,32))
biases1 = np.zeros((32,1))
biases2 = np.zeros((10,1))
x = div(x)
x2 = div(x2)
 
####################################################################################################
#################### Defining Sigmoid activation function and its derivative #######################
#################################################################################################### 
def sigma(x):
  try:
    return 1/(1+(math.exp(-x)))                                    # Sigmoid function
  except:
    return 0.000001                                                # if the value returned by the function is too small, just return a given number
sig = np.vectorize(sigma)                                          # converting function to one that can be applied on a numpy array

def derivative(x):
  try:
    return math.exp(-x)/((1+math.exp(-x))**2)                      # Sigmoid function derivative
  except:
    return 0.000001                                                # if the value returned by the function is too small, just return a given number
deriv = np.vectorize(derivative)                                   # converting function to one that can be applied on a numpy array

####################################################################################################
###################### Defining Relu activation function and its derivative ########################
#################################################################################################### 
def Relu(x):
  return max(0,x)                                                  # ReLu function
ReLu = np.vectorize(Relu)                                          # converting function to one that can be applied on a numpy array

def dRelu(x):                                                      # ReLu function derivative
  if x <=0:
    return 0                                                       
  else:
    return 1

####################################################################################################
############################# Initializing weights and biases randomly #############################
####################################################################################################
def createParams():
    w1 = np.random.uniform(low=-0.1, high=0.1, size = (32,784))
    b1 = np.random.uniform(low=-0.1, high=0.1, size = (32,1))
    w2 = np.random.uniform(low=-0.1, high=0.1, size = (10,32))
    b2 = np.random.uniform(low=-0.1, high=0.1, size = (10,1))
    return w1, b1, w2, b2

####################################################################################################
################################## Defining dropout function here ##################################
####################################################################################################
def dropout(layer, probability):
  i = 0
  numbers = []
  number = 0
  x = probability*len(layer)                                       # Find number of neurons to "drop out" of layer
  while i < x:
    number = random.randint(0,len(layer)-1)                        # Randomly pick neurons to drop out
    if not(number in numbers):                                     # If neuron has already been chosen, don't drop out again 
      layer[number][0] = 0                                         # Applying dropout
      i += 1
    else:
      continue

####################################################################################################
######################################## Forward propogation #######################################
####################################################################################################
def forward(x, w1, b1, w2, b2, y):
    expected = np.zeros((10,1))                                   # Defining expected output array
    expected[y][0] = 1                                            # Making value at right index 1 (ex: If correct digit were 2, array would be [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    x = np.reshape(x, (784,1))     
    layer1 = np.add(np.dot(w1, x), b1)                            # Getting first layer w/ weights and biases                           
    l1 = sig(layer1)                                              # Applying sigmoid to first layer
    layer2 = np.add(np.dot(w2, l1), b2)                           # Getting output layer w/ weights and biases
    l2 = sig(layer2)                                              # Applying sigmoid to output layer
    loss = 0
    w11 = np.square(w1)
    w22 = np.square(w2)
    for i in range(len(l2)):                                      # Computing cost function with quadratic formula
      loss += (l2[i][0] - expected[i][0]) ** 2
    return layer1, l1, layer2, l2, loss
    expected[y] = 0

####################################################################################################
########################################## Backpropogation #########################################
####################################################################################################
def backprop(x, l1, layer1, w1, l2, layer2, w2, y):
    dW1 = np.zeros((32,784))                                      # Array that stores derivative of weights between input and hidden layer
    dB1 = np.zeros((32,1))                                        # Array that stores derivative of biases for hidden layer
    dW2 = np.zeros((10,32))                                       # Array that stores derivative of weights between hidden and output layer
    dB2 = np.zeros((10,1))                                        # Array that stores derivative of biases for output layer
    x = np.reshape(x, (784,1))
    expected = np.zeros((10,1))
    expected[y] = 1
    #print(expected)
    #time.sleep(1)
    dW2 = np.dot(((2*(l2-expected)) * deriv(layer2)), l1.T)       # DCost WRT second weights array = dCost x dSigma x Hidden layer values  
    dB2 = 2*(l2-expected) * deriv(layer2)                         # DCost WRT second biases array = dCost x dSigma
    dActivations = np.dot(w2.T, (2*(l2-expected) * deriv(l2)))    # DCost WRT derivative of activations = dCost x dSigma x Second weights layer
    dW1 = np.dot(x, (dActivations * deriv(layer1)).T)             # DCost WRT first weights array = dActivations x dSigma x Input layer
    dB1 = dActivations*deriv(layer1)                              # DCost WRT first biases array = dActivations x dSigma
    #print(dB1)
    #print(dB2)
    #print("")
    return dW1, dW2, dB1, dB2                                     # Return derivatives

####################################################################################################
############################## Updating Parameters With Derivatives ################################
####################################################################################################
def updateParams(w1, w2, b1, b2, dW1, dW2, dB1, dB2, scale):
  w1 = w1 - dW1.T*scale                                           # Update first weights layer
  w2 = w2 - dW2*scale                                             # Update second weights layer
  b1 = b1 - dB1*scale                                             # Update first biases layer
  b2 = b2 - dB2 * scale                                           # Update second biases layer
  return w1, w2, b1, b2                                           # Return updated weights and biases

####################################################################################################
####################### Getting actual digit prediction from output layer ##########################
####################################################################################################
def getPredictions(output):
  ma = output[0][0]
  index = 0
  for i in range(1, len(output)):
    if output[i][0] > ma:
      ma = output[i][0]
      index = i
  return index

####################################################################################################
############################## Getting accuracy of neural network ##################################
####################################################################################################
def getAccuracy(outputs, y):
  correct = 0
  for i in range(len(outputs)):
    if outputs[i] == y[i]:
      correct += 1
  return correct/len(y)

####################################################################################################
####################################### Gradient Descent ###########################################
####################################################################################################
def gradientDescent(x, y, iterations):                                              # 
  layers = []
  w1, b1, w2, b2 = createParams()
  cost = 0
  accuracy = 0.1
  accuracies = []
  truth = True
  while truth == True:
    predictions = []
    predictions2 = []
    expecteds2 = []
    expected = []
    x3 = np.zeros((784,3500))
    y3 = []
    xd = []
    randoms = []
    for i in range(3500):
      layer1, l1, layer2, l2, cost = forward(x[:,i], w1, b1, w2, b2, y[i][0])
      predictions.append(getPredictions(l2))
      expected.append(y[i][0])
      dW1, dW2, dB1, dB2 = backprop(x[:,i], l1, layer1, w1, l2, layer2, w2, y[i])
      w1, w2, b1, b2 = updateParams(w1, w2, b1, b2, dW1, dW2, dB1, dB2, 0.01)
    for i in range(2030):
        layer1, l1, layer2, l2, cost = forward(x2[:,i], w1, b1, w2, b2, y2[i][0])
        #print(l2)
        #print("")
        predictions2.append(getPredictions(l2))
        expecteds2.append(y2[i][0])
    accuracy2 = getAccuracy(predictions2,expecteds2)
    accuracies.append(accuracy2)
    accuracy = getAccuracy(predictions,expected)
    print("Iteration #" + str(j))
    print("Accuracy On Training:", accuracy)
    print("Accuracy On Test:", accuracy2)
    print("")
    # Checks if test set accuracy on this iteration is larger than accuracy on previous iteration. If so, continue training. If not, stop training.
    if len(accuracies) > 1:
      if accuracies[len(accuracies)-1] < accuracies[len(accuracies)-2]:
        truth = False
  return w1, w2, b1, b2

# Running gradientDescent and saving trained weights & biases in files
weights1, weights2, biases1, biases2 = gradientDescent(x, y, 3500)
np.savetxt('weights3.txt', weights1, delimiter=',')
np.savetxt('weights4.txt', weights2, delimiter=',')
np.savetxt('biases3.txt', biases1, delimiter=',')
np.savetxt('biases4.txt', biases2, delimiter=',')

print("Training Complete!")
print("")
print("")

# One final check for accuracy on test set
predictions2 = []
expecteds2 = []
for i in range(2030):
    layer1, l1, layer2, l2, cost = forward(x2[:,i], weights1, biases1, weights2, biases2, y2[i][0])
    #print(l2)
    #print("")
    predictions2.append(getPredictions(l2))
    expecteds2.append(y2[i][0])
accuracy = getAccuracy(predictions2,expecteds2)
print("Accuracy:", accuracy)
print("")

# Running the neural network! Yay, you made it!     
while True:
  i = input("Enter number: ")
  print("")
  i = int(i)
  layer1 = np.zeros((32,1))
  l1 = np.zeros((32,1))
  layer2 = np.zeros((10,1))
  l2 = np.zeros((10,1))
  cost = 0
  showNum(x2,i)
  print("Correct value:", y2[i][0])
  layer1, l1, layer2, l2, cost = forward(x2[:,i], weights1, biases1, weights2, biases2, y2[i][0])
  print("")
  for k in range(len(l2)):
      print("Confidence that digit is", str(k) + ":", l2[k][0])
  print("Final Prediction:", getPredictions(l2))
  print("")
  print("")
