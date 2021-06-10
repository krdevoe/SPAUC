import numpy as np
import pandas as pd
import random
import math
import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##Read in dataset as a pandas dataframe, called 'df'. The minority class should be class 0, majority class 1. The class column should be titled 'Class'.
df =
#Scale data so each feature has mean 0 and standard deviation 1. Split data into class 0 and class 1.
scaler.fit(df.drop('Class', axis = 1))
data_0 = scaler.transform(df_0.drop('Class', axis = 1))
data_1 = scaler.transform(df_1.drop('Class', axis = 1))


# computes C, where step size at time t is given by n(t) = 1/(2Ct^(theta))
def C(df, lamb):
    scaler.fit(df)
    df = scaler.transform(df)
    k_sq = max(1, 16*max(np.square(df).sum(axis=1)))
    return max(4*lamb, k_sq)

## Code for SPAUC algorithm:

class SPAUC:

    def __init__(self, feat, lamb, C, theta):
        self.n_p = 0
        self.n_n = 0
        self.feat = feat
        self.s_p = np.zeros(self.feat)
        self.s_n = np.zeros(self.feat)
        self.w = np.zeros(self.feat)
        self.t = 0
        self.lamb = lamb
        self.C = C
        self.theta = theta

    def grad(self, x, y, p, u, v):
        if y == 1:
            return (2 * (1 - p) * sum((x - u) ** 2) * self.w + 2 * p * (1 - p) * (v - u) + 2 * p * (1 - p) * sum(
                (u - v) ** 2) * self.w)
        else:
            return (2 * p * sum((x - v) ** 2) * self.w + 2 * p * (1 - p) * (v - u) + 2 * p * (1 - p) * sum(
                (u - v) ** 2) * self.w)

    def step(self):
        return 1 / (2 * self.C * (self.t + 1) ** self.theta)

    def new_w(self, gradient):
        return ((self.w - self.step() * gradient) / (2 * self.lamb * self.step() + 1))

    def f(self, x):
        return sum(self.w * x)

    def optimize(self, feat_x, y):
        for j in range(len(feat_x)):
            if y[j] == 1:
                self.n_p += 1
                self.s_p += feat_x[j,]
            else:
                self.n_n += 1
                self.s_n += feat_x[j,]

            if self.n_p == 0:
                u = np.zeros(self.feat)
            else:
                u = self.s_p / self.n_p
            if self.n_n == 0:
                v = np.zeros(self.feat)
            else:
                v = self.s_n / self.n_n

            ind = random.choices(range(len(feat_x)), k=1)
            g = self.grad(feat_x[ind,], y[ind], self.n_p / (self.t + 1), u, v)
            self.w = self.new_w(g)
            self.t += 1
        return (self.w)

    def AUC(self, X_test, y_test):
        pred = []
        for i in range(len(X_test)):
            pred.append(self.f(X_test[i,]))
        return roc_auc_score(y_test, pred)

##Cross-validation on lambda and theta.

lamb = [0, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000]
theta = [0.5, 0.75, 1]
auc = np.zeros([len(lamb), len(theta)])
x = #controls the number of samples to run in cross-validation
y = #number of features in dataset
for k in range(x):

    X_train0, X_test0, y_train0, y_test0 = train_test_split(data_0, np.zeros(len(data_0)), test_size=.3)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_1, np.ones(len(data_1)), test_size=.3)
    X_train = np.concatenate([X_train0, X_train1])
    y_train = np.concatenate([y_train0, y_train1])
    X_test = np.concatenate([X_test0, X_test1])
    y_test = np.concatenate([y_test0, y_test1])

    #randomize
    ind = random.choices(range(len(X_train)), k=len(X_train))
    X_train = X_train[ind,]
    y_train = y_train[ind]

    for i in range(len(lamb)):
        C1 = C(df.drop('Class', axis=1), lamb[i])
        for j in range(len(theta)):
            s = SPAUC(y, lamb[i], C1, theta[j])
            a = [0]
            while True:
                s.optimize(X_train, y_train)
                a.append(s.AUC(X_train, y_train))
                if abs(a[-1] - a[-2]) < .001:
                    break
            auc[i,j] += s.AUC(X_test, y_test)
auc = auc / x

print(auc)

#Select the best parameters for lambda and theta
lamb =
theta =

#Computes C for use in the step size
C1 = C(df.drop('Class', axis=1), lamb)

x = #number of times to split data into training and test sets.
y = #number of features in dataset.
auc = np.zeros(x)
for k in range(x):

    X_train0, X_test0, y_train0, y_test0 = train_test_split(data_0, np.zeros(len(data_0)), test_size=.3)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_1, np.ones(len(data_1)), test_size=.3)
    X_train = np.concatenate([X_train0, X_train1])
    y_train = np.concatenate([y_train0, y_train1])
    X_test = np.concatenate([X_test0, X_test1])
    y_test = np.concatenate([y_test0, y_test1])

    # randomize
    ind = random.choices(range(len(X_train)), k=len(X_train))
    X_train = X_train[ind,]
    y_train = y_train[ind]

    s = SPAUC(y, lamb, C1, theta)
    a = [0]
    while True:
        s.optimize(X_train, y_train)
        a.append(s.AUC(X_train, y_train))
        if abs(a[-1] - a[-2]) < .001:
            break
    auc[k] = s.AUC(X_test, y_test)

print(auc)
