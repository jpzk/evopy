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

This algorithm is based on "Towards Non-linear Constraint Estimation for
Expensive Optimization" by Fabian Gieseke and Oliver Kramer. The algorithm
might be superior at linear constraints.
'''

from numpy import matrix, transpose, array, eye, ones, exp
from numpy.random import normal, triangular
from numpy.linalg import eigh, norm

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import *
from sys import exit
from multiprocessing import cpu_count
from copy import deepcopy

from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.helper.logger import Logger

class AHMCE2(object):
    def __init__(self, initial, feasible, infeasible, toadd, bt, budget):
        self._initial = initial
        self._feasible = feasible
        self._infeasible = infeasible
        self._toadd = toadd
        self._spent_budget = 0
        self._budget = budget
        self._bt = bt
        self._scaling = ScalingStandardscore()
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

    def _generate_individual(self):
        direction = self._nearest_infeasible - self._mean
        scalar = triangular(-1.0, 0.0, 1.0)
        return matrix(self._mean + scalar * direction)

    # ask if prediction accuracy is good enough.
    def ask_prediction_possible(self, individual):
        return True

    def is_feasible(self, individual):
        y = self._model.predict(self._scaling.scale(individual))
        return True if y == 1 else False

    # tell meta model to train problem
    def tell_train(self, problem):

        self._mean = 0.5 * (self._feasible + self._infeasible)
        self._var = norm(self._feasible - self._infeasible)
        self._spent_budget = 0

        evaluate_cf = lambda x : problem.is_feasible(x)
        encode = lambda b : 1 if b == True else -1

        C_range = [10000]
        tuned_parameters = [{
            'kernel': ['linear'],
            'C': C_range}]

        # spawn initial population
        # we need both feasible and infeasible solutions
        self._X = [self._generate_individual()\
            for i in xrange(self._initial)]
        self._Y = map(encode, map(evaluate_cf, self._X))
        self._spent_budget += len(self._Y)

        # ensure balance, bt equals balance tolerance
        while(abs(sum(self._Y)) > self._bt):
            if(self._spent_budget > self._budget):
                print "Failed to initialize because of balancing"
                exit(0)
            if(sum(self._Y) > self._bt): # more feasibles than infeasible
                x = self._generate_individual()
                if(not problem.is_feasible(x)):
                    self._X.append(x)
                    self._Y.append(encode(False))
                self._spent_budget += 1
            else: # sum(Y) <= bt
                x = self._generate_individual()
                if(problem.is_feasible(x)):
                    self._X.append(x)
                    self._Y.append(encode(True))
                self._spent_budget += 1

        grid = GridSearchCV(SVC(), tuned_parameters,\
        cv = LeaveOneOut(len(self._X)), verbose=0)

        Xa, Ya = map(lambda x : x.getA1(), self._X), array(self._Y)
        self._scaling.setup(Xa)
        Xa = map(lambda x : x.getA1(), map(lambda x : self._scaling.scale(x), Xa))

        grid.fit(Xa, Ya)
        best_C = grid.best_estimator.C
        model = SVC(kernel = 'linear', C = best_C)
        model.fit(Xa, Ya)

        # get the nearest feasible
        tuples = zip(self._X, self._Y)
        geta1 = lambda x : x[0].getA1()

        feasibles = map(geta1, filter(lambda x : x[1] == 1, tuples))
        infeasibles = map(geta1, filter(lambda x : x[1] == -1, tuples))

        feasiblesD = [(x, abs(model.decision_function(x))) for x in feasibles]
        infeasiblesD = [(x, abs(model.decision_function(x))) for x in infeasibles]

        self._nearest_feasible = sorted(feasiblesD, key = lambda t : t[1])[0][0]
        self._nearest_infeasible = sorted(infeasiblesD, key = lambda t : t[1])[0][0]

        # update mean, var
        self._mean = 0.5 * (self._nearest_feasible + self._nearest_infeasible)
        self._var = norm(self._nearest_feasible - self._nearest_infeasible)

        while(self._spent_budget < self._budget):
            # train svm and add new points

            # generate points, check if near the decision boundary
            # use temporary set Xt
            Xt = [self._generate_individual()\
                    for i in xrange(self._toadd)]

            Xta = map(lambda x : (self._scaling.scale(x.getA1())).getA1(), Xt)
            XtaD = [(x, abs(model.decision_function(x))) for x in Xta]
            XtaDs = sorted(XtaD, key = lambda t : t[1])

            # take the solution with smallest distance to hyperplane
            # and evaluate the cf, in the end, append to trainingset
            x = self._scaling.descale(XtaDs[0][0])
            y = encode(problem.is_feasible(x))
            self._spent_budget += 1

            self._X.append(x)
            self._Y.append(y)

            grid = GridSearchCV(SVC(), tuned_parameters,\
            cv = LeaveOneOut(len(self._X)), verbose=0)

            Xa, Ya = map(lambda x : x.getA1(), self._X), array(self._Y)
            self._scaling.setup(Xa)
            Xa = map(lambda x : x.getA1(), map(lambda x : self._scaling.scale(x), Xa))

            grid.fit(Xa, Ya)
            best_C = grid.best_estimator.C
            model = SVC(kernel = 'linear', C = best_C)
            model.fit(Xa, Ya)

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

            self.logger.log()

            print "train ahmce2", sum(self._Y), len(self._X), self._var

        self.trained = True
        self._model = model


