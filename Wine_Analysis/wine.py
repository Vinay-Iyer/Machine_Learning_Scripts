#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:58:29 2017

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

data = pd.read_csv('wine.dat',header=None)
data.columns = ['Class','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium','Total Phenols', \
             'Flavanoids','Nonflavanoid phenols','Proanthocyanins', 'Color Intensity','Hue', 'OD280/OD315 of diluted wines','Proline']

data.head()
data.describe()

# extract explanatory and response variable. 
X = data.loc[:,'Alcohol']
Y = data.loc[:,'Ash']

# Split into training and test
X_train = X[:-50]
X_test  = X[-50:]

# Split the response variables into training/testing sets
Y_train  = Y[:-50]
Y_test   = Y[-50:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train.to_frame(), Y_train.to_frame())

#Predict
regr.predict(X_test.to_frame())

# The coefficients
print('Coefficients: \n', regr.coef_)

# Mean squared error
print("Mean Squared Error: %.2f" \
      %np.mean((regr.predict(X_test.to_frame() - X_test.to_frame()) ** 2)))

# Explain Variance
print('Variance score: %.2f' % regr.score(X_test.to_frame(), Y_test.to_frame()))

# Plot Outputs
plt.scatter(X_test.to_frame(),Y_test.to_frame(),color='red')
plt.plot(X_test.to_frame(), regr.predict(X_test.to_frame()), color='blue')
plt.xticks(())
plt.yticks(())

# -----------------------------------------------------------------------------
# Regression using tensorflow. 


# Initialize Parameters
n_samples = X_train.shape[0]
learning_rate = 0.001
training_epochs = 100
display_step = 1


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
        for (x, y) in zip(X_train, Y_train):
            sess.run(optimizer, feed_dict={Xp: x, Yp: y})    

    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        c = sess.run(cost,feed_dict={Xp: X_train, Yp:Y_train})
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
         "W=", sess.run(W), "b=", sess.run(b) )
        epoch = epoch + 1
                 
    
    print ("Optimization Finished")
    training_cost = sess.run(cost, feed_dict={Xp:X_train, Yp:Y_train})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b) )        

# Plots
    plt.plot(X_train, Y_train, 'ro', label='Original data')
    plt.plot(X_train, sess.run(W) * Y_train + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show() 
