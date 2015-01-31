# 0.97986, for svc.
# 0.98214, for nusvc
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from skimage import measure ,morphology
from numpy import *

# Read in data
train = pd.read_csv('train.csv')
y = train.label
train.drop('label', axis=1, inplace=True)

X = train.values
y = y.values

# Preprocessing
for ind in range(X.shape[0]):
	X[ind,:] = where(X[ind,:] > mean(X[ind,:]),0.,1.)

# PCA
pca = PCA(50) 
pca.fit(X)
X50 = pca.transform(X)

# SVC
svc = svm.SVC(cache_size=2000)
svc.fit(X50,y)

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

# Predict
p = nusvc.predict(test)

# Output submission file
pd.Series(p,range(1,p.size+1)).to_csv('nusvc.csv')



