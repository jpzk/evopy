# Using the magic encoding
# -*- coding: utf-8 -*-

'''
This file is part of evopy.

Copyright 2012 - 2013, Jendrik Poloczek

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

from numpy import matrix, transpose, array, eye, ones, exp
from numpy.random import normal, triangular
from numpy.linalg import eigh, norm

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import *

from evopy.helper.logger import Logger

class LAHMCE(object):
    def __init__(self, feasible, infeasible, toadd, budget, accuracy):
        self._feasible = feasible
        self._infeasible = infeasible
        self._toadd = toadd
        self._spent_budget = 0
        self._budget = budget
        self._model = None
        self._X = None
        self._Y = None
        self._accuracy = accuracy
        self.trained = False
        self.logger = Logger(self)

        self._nearest_feasible = feasible
        self._nearest_infeasible = infeasible

        # define logger bindings
        self.logger.add_binding('_spent_budget', 'spent_budget')
        self.logger.add_binding('_X', 'X')
        self.logger.add_binding('_Y', 'Y')
        self.logger.add_binding('_var', 'var')

    def _generate_individual(self):
        ldirection = self._nearest_infeasible - self._mean
        rdirection = self._nearest_feasible - self._mean
        scalar = triangular(-1.0, 0.0, 1.0)
        if(scalar < 0):
            return matrix(self._mean + abs(scalar) * rdirection)
        if(scalar >= 0):
            return matrix(self._mean + abs(scalar) * ldirection)

    # tell meta model to train problem
    def train(self, problem):

        self._mean = 0.5 * (self._feasible + self._infeasible)
        self._var = norm(self._feasible - self._infeasible)
        self._spent_budget = 0

        evaluate_cf = lambda x : problem.is_feasible(x)
        encode = lambda b : 1 if b == True else -1

        # initial model
        self._X = [self._nearest_feasible, self._nearest_infeasible]
        self._Y = [1, -1]
        model = SVC(kernel = 'linear', C = 10000)
        X = [array(self._nearest_feasible.getA1()), array(self._nearest_infeasible.getA1())]
        Y = array([1, -1])
        model.fit(X, Y)

        while(self._spent_budget < self._budget):
            # train svm and add new points

            # generate points, check if near the decision boundary
            # use temporary set Xt
            Xt = [self._generate_individual()\
                    for i in xrange(self._toadd)]

            Xta = map(lambda x : x.getA1(), Xt)
            XtaD = [(x, abs(model.decision_function(x))) for x in Xta]
            XtaDs = sorted(XtaD, key = lambda t : t[1])

            # take the solution with smallest distance to hyperplane
            # and evaluate the cf, in the end, append to trainingset
            x = XtaDs[0][0]
            y = encode(problem.is_feasible(x))
            self._spent_budget += 1

            self._X.append(x)
            self._Y.append(y)

            model = SVC(kernel = 'linear', C = 10000)
            model.fit(X, Y)

            # update mean, var
            if(y == 1):
                cmp_distance = norm(self._nearest_infeasible - x)
            else:
                cmp_distance = norm(self._nearest_feasible - x)

            if(cmp_distance < self._var):
                if(y == 1):
                    self._nearest_feasible = x
                else:
                    self._nearest_infeasible = x
                self._mean = 0.5 *\
                        (self._nearest_feasible + self._nearest_infeasible)
                self._var = norm(self._nearest_feasible - self._nearest_infeasible)
                if(self._var < self._accuracy):
                    break

            self.logger.log()

            print "train lahmce", sum(self._Y), len(self._X), self._var

        print "mean", self._mean
        print "nearest_feasible_vec", self._nearest_feasible - self._mean
        print "nearest_infeasible_vec", self._nearest_infeasible - self._mean
        print "distance", self._var
        return self._mean, self._nearest_feasible, self._nearest_infeasible, self._spent_budget

