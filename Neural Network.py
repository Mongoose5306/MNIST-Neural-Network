import numpy as np

import time
import os
import math
import time
import random

def convert(num):
  exp = 0
  ee = 0
  for i in range(len(num)):
    if num[i] == 'e':
      ee = i
      for i in range(len(num)):
        if num[i] == '+' or num[i] == '-':
          exp = num[i:len(num)]
  finalstring = ''
  aye = 0
  if int(exp) < 0:
    zero = '.'
    for i in range(len(num)):
      if num[i] == '.':
        aye = i
    for i in range((-1*int(exp)) - 1):
      zero += "0" 
    finalstring = zero+num[aye-1]+num[aye+1:ee]
    return finalstring
  elif int(exp)>0:
      newstring = num[0:ee]
      if len(num[0:ee])-2 > int(exp):
        finalstring = str(num[0]) + str(num[2:2+int(exp)]) + '.' + str(num[2+int(exp):ee])
      elif len(num[0:ee])-2 < int(exp):
        zero = ""
        for i in range(int(exp)-(len(newstring)-(2))):
          zero += "0"
        finalstring = num[0] + num[2:ee] + zero
      elif len(newstring)-2 == int(exp[1:len(exp)]):
        finalstring = num[0] + num[2:ee]
      return finalstring


os.system('clear')
j = 0
fh = open("mnist.txt")
train_x = []
x = np.zeros((784, 3500))
train_y = []
xd = []
y = []
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
    y.append(xd)
    xd = []
  elif len(li[i]) == 784:
    train_x.append(li[i])

train_x = np.array(train_x)

x = train_x.T




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


def divide(x):
  return x/255
div = np.vectorize(divide)

def divide2(x):
  return x/100
div2 = np.vectorize(divide2)
def showNum(x, i):
  pri = []
  pr = []
  for k in range(28):
    for l in range(28):
      if x[:,i][28*k+l] > 0:
        pr.append("*")
        pr.append(" ")
      else:
        pr.append(" ")
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

weights1 = np.zeros((32,784))
weights2 = np.zeros((10,32))
biases1 = np.zeros((32,1))
biases2 = np.zeros((10,1))


x = div(x)
x2 = div(x2)

def sigma(x):
  try:
    return 1/(1+(math.exp(-x)))
  except:
    return 0.000001
sig = np.vectorize(sigma)

def derivative(x):
  try:
    return math.exp(-x)/((1+math.exp(-x))**2)
  except:
    return 0.000001
deriv = np.vectorize(derivative)

def Relu(x):
  return max(0,x)
ReLu = np.vectorize(Relu)

def dRelu(x):
  if x <=0:
    return 0
  else:
    return 1

def createParams():
  #start with very tiny range
    w1 = np.random.uniform(low=-0.1, high=0.1, size = (32,784))
    b1 = np.random.uniform(low=-0.1, high=0.1, size = (32,1))
    w2 = np.random.uniform(low=-0.1, high=0.1, size = (10,32))
    b2 = np.random.uniform(low=-0.1, high=0.1, size = (10,1))
    return w1, b1, w2, b2

def dropout(layer, probability):
  i = 0
  numbers = []
  number = 0
  x = probability*len(layer)
  while i < x:
    number = random.randint(0,len(layer)-1)
    if not(number in numbers):
      numbers.append(number)
      layer[number][0] = 0
      i += 1
    else:
      continue

def forward(x, w1, b1, w2, b2, y):
    expected = np.zeros((10,1))
    expected[y][0] = 1
    x = np.reshape(x, (784,1))
    #x = dropout(x, 0.8)
    #print(w1)
    layer1 = np.add(np.dot(w1, x), b1)
    #print(layer1)
    l1 = sig(layer1)
    #l1 = dropout(l1, 0.5)
    layer2 = np.add(np.dot(w2, l1), b2)
    #print(layer2)
    l2 = sig(layer2)
    loss = 0
    w11 = np.square(w1)
    w22 = np.square(w2)
    for i in range(len(l2)):
      loss += (l2[i][0] - expected[i][0]) ** 2
    return layer1, l1, layer2, l2, loss
    expected[y] = 0

def backprop(x, l1, layer1, w1, l2, layer2, w2, y):
    dW1 = np.zeros((32,784))
    dB1 = np.zeros((32,1))
    dW2 = np.zeros((10,32))
    dB2 = np.zeros((10,1))
    x = np.reshape(x, (784,1))
    expected = np.zeros((10,1))
    expected[y] = 1
    #print(expected)
    #time.sleep(1)
    dW2 = np.dot(((2*(l2-expected)) * deriv(layer2)), l1.T)
    dB2 = 2*(l2-expected) * deriv(layer2)
    dActivations = np.dot(w2.T, (2*(l2-expected) * deriv(l2)))
    dW1 = np.dot(x, (dActivations * deriv(layer1)).T)
    dB1 = dActivations*deriv(layer1)
    #print(dB1)
    #print(dB2)
    #print("")
    return dW1, dW2, dB1, dB2


def updateParams(w1, w2, b1, b2, dW1, dW2, dB1, dB2, scale):
  w1 = w1 - dW1.T*scale
  w2 = w2 - dW2*scale
  b1 = b1 - dB1*scale
  b2 = b2 - dB2 * scale
  #print(w1)
  #print(w2)
  #print(b1)
  #print(b2)
  #print("")
  return w1, w2, b1, b2

def getPredictions(output):
  ma = output[0][0]
  index = 0
  for i in range(1, len(output)):
    if output[i][0] > ma:
      ma = output[i][0]
      index = i
  return index

def getAccuracy(outputs, y):
  correct = 0
  for i in range(len(outputs)):
    if outputs[i] == y[i]:
      correct += 1
  return correct/len(y)

def gradientDescent(x, y, iterations):
  outputs = []
  firstlayer = np.zeros((10,1))
  costs = []
  layers = []
  expected = []
  w1, b1, w2, b2 = createParams()
  cost = 0
  costa = 1
  j = 0
  accuracy = 0.1
  a = 0
  abc = 0
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
      costs.append(cost)
      predictions.append(getPredictions(l2))
      expected.append(y[i][0])
      dW1, dW2, dB1, dB2 = backprop(x[:,i], l1, layer1, w1, l2, layer2, w2, y[i])
      w1, w2, b1, b2 = updateParams(w1, w2, b1, b2, dW1, dW2, dB1, dB2, 0.01)
    costa = sum(costs)/len(costs)
    costs = []
    j+=1
    for i in range(2030):
        layer1, l1, layer2, l2, cost = forward(x2[:,i], w1, b1, w2, b2, y2[i][0])
        #print(l2)
        #print("")
        predictions2.append(getPredictions(l2))
        expecteds2.append(y2[i][0])
    accuracy2 = getAccuracy(predictions2,expecteds2)
    accuracies.append(accuracy2)
    if j % 1 == 0:
        accuracy = getAccuracy(predictions,expected)
        print("Iteration #" + str(j))
        print("Accuracy On Training:", accuracy)
        print("Accuracy On Test:", accuracy2)
        print("")
        #print(costa)
        abc += 1
    if len(accuracies) > 1:
      if accuracies[len(accuracies)-1] < accuracies[len(accuracies)-2]:
        truth = False
  return w1, w2, b1, b2

weights1, weights2, biases1, biases2 = gradientDescent(x, y, 3500)
np.savetxt('weights3v2.txt', weights1, delimiter=',')
np.savetxt('weights4v2.txt', weights2, delimiter=',')
np.savetxt('biases3v2.txt', biases1, delimiter=',')
np.savetxt('biases4v2.txt', biases2, delimiter=',')

print("Training Complete!")

print("")
print("")

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
