import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics
import sys


def k_fold_cross_validation(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation


labels = []
f = open(sys.argv[1])

# read in the first line as the feature labels
# so you can know what you're removing or adding
labels = f.readline().rstrip().split(",")

## read features into list of lists (data)
## read diagnosis into a list of lists (target)
target = []
data = []
print("The features are:", labels)
for line in f:
    parts = line.split(",")
    data.append([float(p) for p in parts[0:-1]])
    target.append(float(parts[-1]))

## conver to numpy arrays
nptarget = np.array(target)
npdata = np.array(data)

## see how many features and training examples you have
print("You have ", npdata.shape[0], "training instanaces")
print("You have ", npdata.shape[1], "features")


## very, very basic classification with Naive Bayes classifier
gnb = GaussianNB()
scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
print("Baseline classification F1:", np.average(scores))

X = [i for i in range(1,40)]
for training, validation in k_fold_cross_validation(X, K=5):
	npdata_train = npdata[training, :]
	npdata_test = npdata[validation, :]

'''
scores = cross_val_score(SVC(kernel='rbf'), npdata, nptarget, cv=5, scoring='f1')
print("SVM (rbf) classification F1 (without feature selection):", np.average(scores))

## Let's say you only want to use features 3 and 4...
#npdata = npdata[:,1:3]
#scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
#print("Baseline classification F1:", np.average(scores))

'''