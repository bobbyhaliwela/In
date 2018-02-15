#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""


import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Import sklearn module for GaussianNB and accuracy score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create classifier
clf = GaussianNB()

# Train
t0 = time() # Time spent for training
clf.fit(features_train, labels_train) # Fit the classifier on the training features and labels
print "\ntraining time:", round(time()-t0, 3), "s"

# Predict
t0 = time() # Time spent for predicting the results
pred = clf.predict(features_test)
print "\npredict time:", round(time()-t0, 3), "s"

# Results
accuracy = accuracy_score(pred, labels_test)
print '\naccuracy = {0}'.format(accuracy)


#########################################################


