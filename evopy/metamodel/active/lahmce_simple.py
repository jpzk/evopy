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
from numpy.linalg import eigh, svd, norm

from evopy.helper.logger import Logger

class LAHMCESimple(object):
    def __init__(self, feasible, infeasible, toadd, budget):
        self._feasible = feasible
        self._infeasible = infeasible
        self._toadd = toadd
        self._spent_budget = 0
        self._budget = budget
        self._model = None
        self._X = None
        self._Y = None
        self.trained = False
        self.logger = Logger(self)

        self._nearest_feasible = feasible
        self._nearest_infeasible = infeasible

        # define logger bindings
        self.logger.add_binding('_spent_budget', 'spent_budget')
        self.logger.add_binding('_X', 'X')
        self.logger.add_binding('_Y', 'Y')
        self.logger.add_binding('_distance', 'distance')

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

        # In direction of self._mean e.g.
        # self._mean -1 negative direction, lies infeasible then -1, False
        # self._mean +1 positive direction, lies feasible then +1, True
        self._positive = True if abs(norm(self._feasible)) > abs(norm(self._mean)) else False
        self._negative = not self._positive

        self._distance = norm(self._feasible - self._infeasible)
        self._spent_budget = 0

        evaluate_cf = lambda x : problem.is_feasible(x)
        encode = lambda b : 1 if b == True else -1

        while(self._spent_budget < self._budget):
            # generate points, check if near the decision boundary
            # use temporary set Xt
            Xt = [self._generate_individual()\
                    for i in xrange(self._toadd)]

            Xta = map(lambda x : x.getA1(), Xt)
            XtaD = [(x, norm(x - self._mean)) for x in Xta]
            XtaDs = sorted(XtaD, key = lambda t : t[1])

            # take the solution with smallest distance to hyperplane
            # and evaluate the cf, in the end, append to trainingset
            x = XtaDs[0][0]
            y = encode(problem.is_feasible(x))
            self._spent_budget += 1

            # update mean, var
            if(y == 1):
                cmp_distance = norm(self._nearest_infeasible - x)
            else:
                cmp_distance = norm(self._nearest_feasible - x)

            if(cmp_distance < self._distance):
                if(y == 1):
                    self._nearest_feasible = x
                else:
                    self._nearest_infeasible = x
                self._mean = 0.5 *\
                        (self._nearest_feasible + self._nearest_infeasible)
                self._distance = norm(self._nearest_feasible - self._nearest_infeasible)

            self.logger.log()

        #print "mean", self._mean
        """
        print "nearest_feasible_vec", self._nearest_feasible - self._mean
        print "nearest_infeasible_vec", self._nearest_infeasible - self._mean
        """
        #print "distance", self._distance
        return self._mean, self._nearest_feasible, self._nearest_infeasible, self._spent_budget

