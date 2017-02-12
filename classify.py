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

labels = []
f = open(sys.argv[1])
testf = open(sys.argv[2])

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

fst = True
test_data = []
filenames = []
for line in testf:
	if not fst:
		parts = line.split(",")
		test_data.append([float(p) for p in parts[1:]])
		filenames.append(parts[0])
	fst = False


## conver to numpy arrays
nptarget = np.array(target)
npdata = np.array(data)
test_npdata = np.array(test_data)

## see how many features and training examples you have
print("You have ", npdata.shape[0], "training instanaces")
print("You have ", test_npdata.shape[0], "testing instanaces")
print("You have ", test_npdata.shape[1], "features")
print("You have ", npdata.shape[1], "features")


## very, very basic classification with Naive Bayes classifier
gnb = GaussianNB()
scores = cross_val_score(gnb, npdata, nptarget, cv=5, scoring='f1')
print("Baseline classification F1:", np.average(scores))

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=2, scoring='f1')
selector = rfecv.fit(npdata, nptarget)

selected_features = []
npdata_sel = np.zeros((npdata.shape[0], selector.n_features_))
test_npdata_sel = np.zeros((test_npdata.shape[0], selector.n_features_))
index = 0
for i, rank in enumerate(selector.ranking_):
	if rank == 1:
		selected_features.append(labels[i])
		npdata_sel[:, index] = npdata[:, i]
		test_npdata_sel[:, index] = test_npdata[:, i]
		index += 1

print("Selected Features (%s)" %(','.join(selected_features)))
print("New feature dimension (%d)" % npdata_sel.shape[1])

scores = cross_val_score(svc, npdata_sel, nptarget, cv=5, scoring='f1')
print("SVM (linear) classification F1 (with feature selection):", np.average(scores))

svc.fit(npdata_sel, nptarget)
prediction = svc.predict(test_npdata_sel)
outfile = open('_output_svm_lin_feat.csv', 'w')
for i, p in enumerate(prediction):
	outfile.write(filenames[i] + "," + str(prediction[i]) + "\n")
outfile.close()


svc = SVC(kernel='rbf')
scores = cross_val_score(svc, npdata, nptarget, cv=2, scoring='f1')
print("SVM (linear) classification F1 (without feature selection):", np.average(scores))

svc.fit(npdata, nptarget)
prediction = svc.predict(test_npdata)
outfile = open('_output_svm_rbf_all.csv', 'w')
for i, p in enumerate(prediction):
	outfile.write(filenames[i] + "," + str(prediction[i]) + "\n")
outfile.close()
