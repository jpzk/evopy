import math
import numpy as np
import pylab as pl
from sklearn import svm

X_feasible = np.random.randn(20, 2) - [5,0]
X_infeasible = np.random.randn(20, 2) + [5, 0]
X = np.r_[X_feasible, X_infeasible]
Y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X, Y)

margin = 1.0/np.sqrt(np.sum(clf.coef_ ** 2))
xx = np.linspace(-20,20)

w = clf.coef_[0]
nw = w / np.sqrt(np.sum(w ** 2))

print "nw", nw
print "b", clf.intercept_[0] / w[1]

for x in X_infeasible:
    s = 2 * (clf.decision_function(x) * (1/np.sqrt(np.sum(w ** 2))))
    nx = x + (nw * s)
    print x, nx, s

