import math
import numpy as np
import pylab as pl
from sklearn import svm

fig = pl.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)

pl.title("Infeasible mirroring with linear kernel")
pl.axis([-10, 10, -10, 10])

np.random.seed(0)
X_feasible = np.random.randn(20, 2) - [2, 2]
X_infeasible = np.random.randn(20, 2) + [4, 3]
X = np.r_[X_feasible, X_infeasible]
Y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel='linear', C = 0.1)
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
a2 = w[1] / w[0]

print "clf.intercept" , clf.intercept_[0]
print "w" , w
margin = 1.0/np.sqrt(np.sum(clf.coef_ ** 2))
xx = np.linspace(-20,20)
yy = a * xx - (clf.intercept_[0]) / w[1]

b = clf.support_vectors_[0] # FEASIBLE
yy_down = yy + a * margin

b = clf.support_vectors_[-1] # INFEASIBLE
yy_up = yy - a * margin 

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

for x in X_infeasible:
    pl.scatter(x[0], x[1], facecolors='red')    
    
 #   s = - ((clf.decision_function(x)) / np.sqrt(np.sum(clf.coef_ ** 2)))
    s = - 2 * (clf.decision_function(x)*(1/np.sqrt(np.sum(clf.coef_ ** 2))))
    print "clf.decision_function(x)", clf.decision_function(x)
    print "margin" , margin

    yy_mirror = a2 * xx - (a2 * x[0] - x[1]) 
    pl.plot(xx, yy_mirror, alpha = 0.25, color= 'gray')
    pl.scatter(x[0] + s, a2 * (x[0] + s) - (a2 * x[0] - x[1]), facecolors='green')

for x in X_feasible: 
    pl.scatter(x[0], x[1], facecolors='gray', alpha=0.5)

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')

pl.show()
