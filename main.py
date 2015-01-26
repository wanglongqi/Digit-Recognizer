import pandas as pd
from sklearn.decomposition import PCA
from sklearn.qda import QDA
from sklearn.metrics import accuracy_score

# Read in data
train = pd.read_csv('train.csv')
y = train.label
train.drop('label', axis=1, inplace=True)

X = train.values
y = y.values

# Normalize
X = X / 255.

# PCA
pca = PCA(50) 
pca.fit(X)
X50 = pca.transform(X)

# QDA
qda = QDA()
qda.fit(X50,y)

# Accuracy score
p = qda.predict(X50)
print 'Accuracy score = ',accuracy_score(p,y)

# Read in test data
test = pd.read_csv('test.csv')

# Normalize
test = test / 255.

# Transform data
test = pca.transform(test)

# Predict
p = qda.predict(test)

# Output submission file
pd.Series(p,range(1,p.size+1)).to_csv('out.csv')
