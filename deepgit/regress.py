import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from enum import IntEnum
import math
import pickle
import torch
import matplotlib.pyplot as plt


with open('data/final/x_encoded.p','rb') as file:
    X = pickle.load(file)
    X = [x.detach().numpy()[:10] for x in X]
with open('data/final/y_stats.p','rb') as file:
    Y_stats = pickle.load(file)

# print (Y_stats)

def plot():
    plt.plot(X)
    plt.show()

def evaluate(model, target, use_log=False):
    Y = [repo[target] for repo in Y_stats]

    if use_log:
        Y = np.log(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    model.fit(X_train, Y_train)
    r2_train = r2_score(Y_train, model.predict(X_train))
    r2_test = r2_score(Y_test, model.predict(X_test))
    print(target)
    print (r2_train, r2_test)

plot()

for target in {'stars','forks', 'score', 'issues'}:
    evaluate(linear_model.LinearRegression(), target)
    evaluate(RandomForestRegressor(n_estimators=100), target)
