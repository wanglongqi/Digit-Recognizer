from sklearn import grid_search
pars = {'nu':[0.0001,0.001,0.002,0.005,0.01,.1],'kernel':['rbf','poly']}
nusvc = svm.NuSVC(cache_size=2000)
gridsearch = grid_search.GridSearchCV(nusvc,pars,cv=5,n_jobs=-1)
gridsearch.fit(X50,y)

# Read in test data
test = pd.read_csv('test.csv')
test = test.values

# Preprocessing
for ind in range(test.shape[0]):
	test[ind,:] = where(test[ind,:] > mean(test[ind,:]),0.,1.)

# Transform data
test = pca.transform(test)


# Predict
p = gridsearch.best_estimator_.predict(test)

# Output submission file
pd.Series(p,range(1,p.size+1)).to_csv('cv.csv')



# kf = KFold(y.size, n_folds=3)
# yp = y*0

# ind = arange(y.size)
# # random.shuffle(ind)
# for train ,test in kf:
#     Xtrain, Xtest, ytrain, ytest = X50[ind[train],:], X50[ind[test],:], y[ind[train]],y[ind[test]]    
#     nusvc.fit(Xtrain,ytrain)
#     print nusvc.score(Xtest,ytest)
#     yp[ind[test]] = nusvc.predict(Xtest)
# from sklearn.metrics import classification_report
# print classification_report(y,yp)
