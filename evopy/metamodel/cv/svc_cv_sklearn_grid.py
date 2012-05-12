''' 
This file is part of evolutionary-algorithms-sandbox.

evolutionary-algorithms-sandbox is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evolutionary-algorithms-sandbox is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evolutionary-algorithms-sandbox.  If not, see <http://www.gnu.org/licenses/>.
'''

from math import floor
import numpy

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

class SVCCVSkGrid():
    """ A strategy for crossvalidation """

    def __init__(self, gamma_range, C_range, cv_method):
        self._gamma_range = gamma_range
        self._C_range = C_range
        self._cv_method = cv_method

    def crossvalidate(self, feasible, infeasible):
        """ This method returns a pair (C, gamma) with classifcation rate
            is maximized. """

        tuned_parameters = [{
            'kernel': ['rbf'], 
            'gamma': self._gamma_range,
            'C': self._C_range}]

        X = numpy.array([f.value for f in feasible] + [i.value for i in infeasible])
        y = numpy.array([1] * len(feasible) + [-1] * len(infeasible))

        clf = GridSearchCV(SVC(), tuned_parameters, cv=self._cv_method)

        clf.fit(X, y)
        best_accuracy = clf.best_score_

        print "best accuracy %f best C %f best gamma %f" %\
            (best_accuracy, clf.best_estimator_.C, clf.best_estimator_.gamma)

        return (feasible,
            infeasible,
            clf.best_estimator_.C,
            clf.best_estimator_.gamma)
