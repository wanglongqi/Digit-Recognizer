# 0.97986, for svc.
# 0.98214, for nusvc
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from skimage import measure ,morphology
from numpy import *
import numpy as np
# add rotate function from
# https://github.com/nicholaslocascio/kaggle-mnist-digits/blob/master/helpers.py
import scipy.ndimage as nd


# Read in data
train = pd.read_csv('train.csv')
y = train.label
train.drop('label', axis=1, inplace=True)

X = train.values
y = y.values


def rotate_dataset(X, Y, n_rotations=2):
    for rot_i in range(n_rotations):
        rot_shape = (X.shape[0], X.shape[1])
        rot_X = np.zeros(rot_shape)
        for index in range(X.shape[0]):
            sign = random.choice([-1, 1])
            angle = np.random.randint(1, 12)*sign
            rot_X[index, :] = (nd.rotate(np.reshape(X[index, :], ((28, 28))), angle, reshape=False).ravel())
        XX = np.vstack((X,rot_X))
        YY = np.hstack((Y,Y))
    return XX, YY


# Preprocessing
for ind in range(X.shape[0]):
	X[ind,:] = where(X[ind,:] > mean(X[ind,:]),0.,1.)

X, y = rotate_dataset(X, y, 10)

# PCA
pca = PCA(50) 
pca.fit(X)
X50 = pca.transform(X)

# SVC
svc = svm.SVC(C=1, cache_size=2000)
svc.fit(X50,y)

# Recall
print accuracy_score(y,svc.predict(X50))

# Read in test data
test = pd.read_csv('test.csv')
test = test.values

# Preprocessing
for ind in range(test.shape[0]):
	test[ind,:] = where(test[ind,:] > mean(test[ind,:]),0.,1.)

# Transform data
test = pca.transform(test)

# Predict
p = svc.predict(test)

# Output submission file
pd.Series(p,range(1,p.size+1)).to_csv('svc.csv')


# NuSVC
nusvc = svm.NuSVC(nu=0.01,cache_size=2000)
nusvc.fit(X50,y)

# Recall
print accuracy_score(y,nusvc.predict(X50))

# Predict
p = nusvc.predict(test)

# Output submission file
pd.Series(p,range(1,p.size+1)).to_csv('nusvc.csv')



