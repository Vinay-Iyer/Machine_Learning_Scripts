#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:25:54 2017

@author: venkatrajgopal
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import datasets, linear_model
import tensorflow as tf

rng = np.random
# Read the data and assign the attributes

data = pd.read_csv('/home/venkatrajgopal/Venkat/Python_Tuts/winedata',header=None)
data.columns = ['Class','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium','Total Phenols', \
             'Flavanoids','Nonflavanoid phenols','Proanthocyanins', 'Color Intensity','Hue', 'OD280/OD315 of diluted wines','Proline']

data.head()
data.describe()

# extract explanatory and response variable. 
X = data.loc[:,'Alcohol']
Y = data.loc[:,'Malic acid']

# Split into training and test
X_train = X[:-50]
X_test  = X[-50:]

# Split the response variables into training/testing sets
Y_train  = Y[:-50]
Y_test   = Y[-50:]

# Regression using tensorflow. 

train_X = np.array(X_train)
train_Y = np.array(Y_train)

# Initialize Parameters
n_samples = train_X.shape[0]
learning_rate = 0.1
training_epochs = 100
display_step = 50


# Set placeholders 
Xp = tf.placeholder("float")
Yp = tf.placeholder("float")

# Set the weight and bias 
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(Xp,W),b)

# Mean Squared error
cost = tf.reduce_sum(tf.pow(pred - Yp, 2)) / (2*n_samples)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the session
with tf.Session() as sess:
    sess.run(init)
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={Xp: x, Yp: y})    

    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        c = sess.run(cost,feed_dict={Xp: train_X, Yp:train_Y})
        print ('Epoch:'), '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
         "W=", sess.run(W), "b=", sess.run(b)
                 
    
    print ('Optimization Finished')
    training_cost = sess.run(cost, feed_dict={Xp:train_X, Yp:train_Y})
    print ("Training cost="), training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'        
    
   
# Plots
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_Y + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show() 
