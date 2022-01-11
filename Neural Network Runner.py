import numpy as np
import time
import math

#early stopping: stop training before 95%
#not how is usually solved in irl tho
#have 3 sets: training set, slightly smaller validation set, and then different test set
#divide training data into 90% training, 10% validation set
#K-Fold Cross Validation, more sophisticated way of doing validation set

#another way to handle overfit: regularization
#force model to not overfit, overfitting happens when weights' magnitude increases too much
#add component to loss function which penalizes big weights
#L2 regularization

#normalization: third way
#input data between 0 and 1, divide by 255

#dropout is not the first thing to try when overfitting, only useful with a huge network
#see what happens when certain weights off, or 0
#https://www.kdnuggets.com/2019/12/5-techniques-prevent-overfitting-neural-networks.html

#use libraries

def forward(x, w1, b1, w2, b2, y):
    expected = np.zeros((10,1))
    expected[y][0] = 1
    x = np.reshape(x, (784,1))
    #print(w1)
    layer1 = np.add(np.dot(w1, x), b1)
    l1 = sig(layer1)
    layer2 = np.add(np.dot(w2, l1), b2)
    l2 = sig(layer2)
    loss = 0
    for i in range(len(l2)):
      loss += (l2[i][0] - expected[i][0]) ** 2
    return layer1, l1, layer2, l2, loss
    expected[y] = 0

def sigma(x):
  try:
    return 1/(1+(math.exp(-x)))
  except:
    return 0.000001
sig = np.vectorize(sigma)

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

weights1 = np.zeros((32,784))

fh = open('weights3.txt')
file = fh.read()

lis = file.split('\n')

for i in range(32):
  lis[i] = lis[i].split(",")
  
for i in range(32):
  for j in range(len(lis[i])):
    weights1[i][j] = lis[i][j]

#print(weights1.shape)
weights2 = np.zeros((10,32))

fh = open('weights4.txt')
file = fh.read()

lis = file.split('\n')

for i in range(10):
  lis[i] = lis[i].split(",")
  
for i in range(10):
  for j in range(len(lis[i])):
    weights2[i][j] = lis[i][j]

#print(weights2.shape)

biases1 = np.zeros((32,1))

fh = open('biases3.txt')
file = fh.read()

lis = file.split('\n')

for i in range(32):
  lis[i] = lis[i].split(",")
  
for i in range(32):
  for j in range(len(lis[i])):
    biases1[i][j] = lis[i][j]

#print(biases1.shape)

biases2 = np.zeros((10,1))

fh = open('biases4.txt')
file = fh.read()

lis = file.split('\n')

for i in range(10):
  lis[i] = lis[i].split(",")
  
for i in range(10):
  for j in range(len(lis[i])):
    biases2[i][j] = lis[i][j]

#print(biases2.shape)

j = 0
fh = open("mnist2.txt")
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

def divide(x):
  return x/255
div = np.vectorize(divide)

x = train_x.T
x = div(x)


predictions2 = []
expecteds2 = []
for i in range(2030):
    layer1, l1, layer2, l2, cost = forward(x[:,i], weights1, biases1, weights2, biases2, y[i][0])
    predictions2.append(getPredictions(l2))
    expecteds2.append(y[i][0])
accuracy = getAccuracy(predictions2,expecteds2)
print("Accuracy:", accuracy)
print("")
prediction = 0
correct = 0
i = 0

import PIL
from PIL import Image
import numpy as np

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

"""x = np.zeros((784,1))
img = PIL.Image.open('Thre.jpg')
pix = img.load()
su = 0
for i in range(28):
    for j in range(28):
        for k in range(3):
            su += pix[j,i][k]
            
        x[28*i+j][0] = (255-su/3)/255
        if x[28*i + j][0] < 0.4:
            x[28*i+j][0] = 0
        su = 0
"""

layer1 = np.zeros((32,1))
l1 = np.zeros((32,1))
layer2 = np.zeros((10,1))
l2 = np.zeros((10,1))
cost = 0

while True:
    num111 = int(input("Enter a number: "))
    showNum(x, num111)
    layer1, l1, layer2, l2, cost = forward(x, weights1, biases1, weights2, biases2, y[num111][0])
    print("")
    for k in range(len(l2)):
        print("Confidence that digit is", str(k) + ":", l2[k][0])
    print("Final Prediction:", getPredictions(l2))
    print("")
    print("")
