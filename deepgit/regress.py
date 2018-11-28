import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math

import pickle

y_f = open('labels.p','rb')
x_f = open('vecs.p','rb')

X = pickle.load(x_f)
Y = pickle.load(y_f)
X = [x.detach().numpy() for x in X]
watchers = [int(math.log10(y[0])) for y in Y]
stargazers = [y[1] for y in Y]

print(len(X),len(watchers))

# x,y,xv,yv = train_test_split(X, watchers,test_size=0.1)

# reg = linear_model.LogisticRegression(penalty='l1', C=0.1, intercept_scaling=1)
# reg.fit(X[:-80], watchers[:-80])

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X[:-80], watchers[:-80])

print([int(p) for p in reg.predict(X[:20])], watchers[:20])

pred = rf.predict(X[-80:])
error = []
for i in range(-80,0):
	error.append(abs(watchers[i]-pred[i])/watchers[i])
print (np.mean(error))



