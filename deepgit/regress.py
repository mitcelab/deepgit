import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math

import pickle

y_f = open('labels.p','rb')
x_f = open('vecs.p','rb')

X = pickle.load(x_f)
Y = pickle.load(y_f)
X = [x.detach().numpy() for x in X]
watchers = [int(math.log10(y[0])) for y in Y]
stargazers = [int(math.log10(y[1])) for y in Y]

watchersl = [y[0] for y in Y]
stargazersl = [y[1] for y in Y]

x,xv,y,yv = train_test_split(X, stargazers,test_size=0.1)
xl,xlv,yl,ylv = train_test_split(X, stargazersl,test_size=0.1)

reg_linear = linear_model.LinearRegression()
reg_linear.fit(x,y)

reg_log = linear_model.LogisticRegression(penalty='l2', C=10)
reg_log.fit(x, y)

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(x, y)

plin = reg_linear.predict(xv)
plog = reg_log.predict(xv)
pred = rf.predict(xv)

print('ERROR')

error = []
for i in range(len(ylv)):
	error.append(abs(ylv[i]-plin[i])/ylv[i])
print (np.mean(error))

error = []
for i in range(len(yv)):
	error.append(abs(yv[i]-plog[i])/yv[i])
print (np.mean(error))

error = []
for i in range(len(yv)):
	error.append(abs(yv[i]-pred[i])/yv[i])
print (np.mean(error))

print('VAL')

print (r2_score(pred, yv))
print (r2_score(plog, yv))
print (r2_score(plin, yv))

print('TRAIN')

print (r2_score(rf.predict(x), y))
print (r2_score(reg.predict(x), y))
print (r2_score(reg_linear.predict(x), y))


