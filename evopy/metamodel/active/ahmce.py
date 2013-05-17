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

This algorithm is presented in "Towards Non-linear Constraint Estimation for
Expensive Optimization" by Fabian Gieseke and Oliver Kramer
'''

from numpy import matrix, transpose, array
from numpy.random import normal
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import *
from sys import exit
from multiprocessing import cpu_count

from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.helper.logger import Logger

class AHMCE(object):
    def __init__(self, mean, var, initial, toadd, bt, budget, kernel):
        self._mean = mean
        self._var = var
        self._initial = initial
        self._toadd = toadd
        self._spent_budget = 0
        self._budget = budget
        self._kernel = kernel
        self._bt = bt
        self._scaling = ScalingStandardscore()
        self._model = None
        self._X = None
        self._Y = None
        self.trained = False
        self.logger = Logger(self)

        # define logger bindings
        self.logger.add_binding('_spent_budget', 'spent_budget')
        self.logger.add_binding('_X', 'X')
        self.logger.add_binding('_Y', 'Y')

    def _generate_individual(self):
        normals = [normal(0.0, d) for d in self._var.getA1().tolist()]
        normals = matrix([normals])
        return self._mean + normals

    # ask if prediction accuracy is good enough.
    def ask_prediction_possible(self, individual):
        return True

    def is_feasible(self, individual):
        self._model.predict(individual)

    # tell meta model to train problem
    def tell_train(self, problem):
        mean = self._mean
        var = self._var
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
            else: # sum(Y) < bt, sum(Y) == bt is impossible
                x = self._generate_individual()
                if(problem.is_feasible(x)):
                    self._X.append(x)
                    self._Y.append(encode(True))
                self._spent_budget += 1

        grid = GridSearchCV(SVC(), tuned_parameters,\
        cv = LeaveOneOut(len(self._X)), n_jobs = cpu_count(), verbose=0)

        Xa, Ya = map(lambda x : x.getA1(), self._X), array(self._Y)

        grid.fit(Xa, Ya)
        best_C = grid.best_estimator.C
        model = SVC(kernel = 'linear', C = best_C)
        model.fit(Xa, Ya)

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
            x = matrix([XtaDs[0][0]])
            y = encode(problem.is_feasible(x))
            self._spent_budget += 1
            self._X.append(x)
            self._Y.append(y)

            grid = GridSearchCV(SVC(), tuned_parameters,\
            cv = LeaveOneOut(len(self._X)), n_jobs = cpu_count(), verbose=0)

            Xa, Ya = map(lambda x : x.getA1(), self._X), array(self._Y)

            grid.fit(Xa, Ya)
            best_C = grid.best_estimator.C
            model = SVC(kernel = 'linear', C = best_C)
            model.fit(Xa, Ya)

            self.logger.log()

            print "train ahmce", sum(self._Y), len(self._X)

        self.trained = True
        self._model = model


