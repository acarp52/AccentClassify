import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys

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

## Let's say you only want to use features 3 and 4...
#npdata = npdata[:,1:3]
#scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
#print("Baseline classification F1:", np.average(scores))

