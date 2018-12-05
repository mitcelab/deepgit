import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from enum import IntEnum
import math
import pickle


with open('vecs.p','rb') as file:
    X = pickle.load(file)
    X = [x.detach().numpy() for x in X]
with open('labels.p','rb') as file:
    Y_all = np.array(pickle.load(file))


class Metric(IntEnum):
    WATCHERS=0
    STARGAZERS=1

print (Y_all)

def evaluate(model, target, use_log=False):
    global Y_all
    Y = Y_all[:,int(target)]

    if use_log:
        Y = np.log(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    model.fit(X_train, Y_train)
    r2_train = r2_score(Y_train, model.predict(X_train))
    r2_test = r2_score(Y_test, model.predict(X_test))
    print(model, target) 
    print (r2_train, r2_test)

# Watchers
evaluate(linear_model.LinearRegression(), target=Metric.WATCHERS)
evaluate(linear_model.LinearRegression(), target=Metric.WATCHERS, use_log=True)
evaluate(RandomForestRegressor(n_estimators=100), target=Metric.WATCHERS)
evaluate(RandomForestRegressor(n_estimators=100), target=Metric.WATCHERS, use_log=True)

# Stargazers
evaluate(linear_model.LinearRegression(), target=Metric.STARGAZERS)
evaluate(linear_model.LinearRegression(), target=Metric.STARGAZERS, use_log=True)
evaluate(RandomForestRegressor(n_estimators=100), target=Metric.STARGAZERS)
evaluate(RandomForestRegressor(n_estimators=100), target=Metric.STARGAZERS, use_log=True)


