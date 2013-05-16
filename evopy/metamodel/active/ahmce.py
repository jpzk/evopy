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

from numpy import matrix, transpose, array
from numpy.random import normal
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import *
from sys import exit
from multiprocessing import cpu_count
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore

class AHMCE(object):
    def __init__(self, problem, initial, toadd, bt, budget, kernel):
        self._problem = problem
        self._initial = initial
        self._toadd = toadd
        self._budget = budget
        self._kernel = kernel
        self._bt = bt
        self._scaling = ScalingStandardscore()

    def _generate_individual(self, mean, var):
        normals = [normal(0.0, d) for d in var.getA1().tolist()]
        normals = matrix([normals])
        return mean + normals

    def train(self, mean, var):
        spent_budget = 0
        evaluate_cf = lambda x : self._problem.is_feasible(x)
        encode = lambda b : 1 if b == True else -1


        # spawn initial population
        # we need both feasible and infeasible solutions
        X = [self._generate_individual(mean, var)\
            for i in xrange(self._initial)]

        Y = map(encode, map(evaluate_cf, X))
        spent_budget += len(Y)

        # ensure balance, bt equals balance tolerance
        while(abs(sum(Y)) > self._bt):
            if(spent_budget > self._budget):
                print "Failed to initialize because of balancing"
                exit(0)
            if(sum(Y) > self._bt): # more feasibles than infeasible
                x = self._generate_individual(mean, var)
                if(not self._problem.is_feasible(x)):
                    X.append(x)
                    Y.append(encode(False))
                spent_budget += 1
            else: # sum(Y) < bt, sum(Y) == bt is impossible
                x = self._generate_individual(mean, var)
                if(self._problem.is_feasible(x)):
                    X.append(x)
                    Y.append(encode(True))
                spent_budget += 1

        while(spent_budget < self._budget):
            # train svm and add new points

            C_range = [10000]
            tuned_parameters = [{
                'kernel': ['linear'],
                'C': C_range}]

            grid = GridSearchCV(SVC(), tuned_parameters,\
                    cv = LeaveOneOut(len(X)), n_jobs = cpu_count(), verbose=0)

            Xa, Ya = map(lambda x : x.getA1(), X), array(Y)

            grid.fit(Xa, Ya)
            best_C = grid.best_estimator.C
            model = SVC(kernel = 'linear', C = best_C)
            model.fit(Xa, Ya)

            # generate points, check if near the decision boundary
            X = [self._generate_individual(mean, var)\
                    for i in xrange(self._toadd)]

            Xa = map(lambda x : x.getA1(), X)
            D = model.decision_function(Xa)
            import pdb
            pdb.set_trace()

            # looking for nearest feasible and infeasible solution

    def is_feasible(self, individual):
        pass
