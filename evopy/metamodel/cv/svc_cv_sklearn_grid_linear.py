''' 
This file is part of evopy.

Copyright 2012, Jendrik Poloczek

evopy is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evopy is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evopy.  If not, see <http://www.gnu.org/licenses/>.
'''

from math import floor
from numpy import array

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

class SVCCVSkGridLinear():
    """ A strategy for crossvalidation """

    def __init__(self, C_range, cv_method):
        self._C_range = C_range
        self._cv_method = cv_method

    def crossvalidate(self, feasible, infeasible):
        """ This method returns a pair (C, gamma) with classifcation rate
            is maximized. """

        tuned_parameters = [{
            'kernel': ['linear'], 
            'C': self._C_range}]

        X = array([f.getA1() for f in feasible] + [i.getA1() for i in infeasible])
        y = array([1] * len(feasible) + [-1] * len(infeasible))

        clf = GridSearchCV(SVC(), tuned_parameters, cv=self._cv_method)

        clf.fit(X, y)
        best_accuracy = clf.best_score_

        return feasible, infeasible, clf.best_estimator_.C, best_accuracy
