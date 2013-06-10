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

from evopy.helper.logger import Logger

from numpy import matrix, transpose, array, dot
from numpy.linalg import norm
from numpy.random import normal

class LBCE(object):

    def __init__(self, feasibles, infeasibles, mean, var, budget):
        self._mean = mean
        self._var = var
        self._spent_budget = 0
        self._budget = budget
        self._X = None
        self._Y = None
        self.feasibles = feasibles
        self.infeasibles = infeasibles
        self.trained = False
        self.logger = Logger(self)

    def _generate_individual(self):
        normals = [normal(0.0, d) for d in self._var.getA1().tolist()]
        normals = matrix([normals])
        return self._mean + normals

    def _binary_search(self, points, problem):
        feasible, infeasible = points
        spent_budget = 0
        while(spent_budget < self._budget):
            if(spent_budget == 0):
                point = feasible + ((infeasible - feasible) / 2)
            else:
                is_feasible = problem.is_feasible(point)
                if(is_feasible):
                    feasible = point
                    point = point + (infeasible - point) / 2
                else:
                    infeasible = point
                    point = feasible + (point - feasible) / 2
            spent_budget += 1
        return point

    def _gram_schmidt(self, plain_vectors):
        v = []
        for i in xrange(len(plain_vectors)):
            if i == 0:
                v.append(plain_vectors[i] / norm(plain_vectors[i]))
            else:
                v.append(0)
                for j in range(i):
                    v[i] += (dot(v[j], plain_vectors[i]) / dot(v[j], v[j])) * v[j]
                v[i] = plain_vectors[i] - v[i]
                v[i] = v[i] / norm(v[i])
        return array(v)

    def tell_train(self, problem):
        mean = self._mean
        var = self._var
        self._spent_budget = 0
        evaluate_cf = lambda x : problem.is_feasible(x)
        encode = lambda b : 1 if b == True else -1

        # spawn initial population
        # we need both feasible and infeasible solutions
        # exactly 1 + d points, 1 feasible and d infeasible

        feasible = self.feasibles[0]
        infeasibles = self.infeasibles

        # for each line
        nearest = []
        plain_vectors = []
        for inf in infeasibles:
            points = (feasible, inf)
            nearest.append(self._binary_search(points, problem))

        for point in nearest[1:]:
            plain_vectors.append((point - nearest[0]).getA1().tolist())
        plain_vectors.append((feasible - infeasibles[0]).getA1().tolist())

        gram_schmidt_vecs = self._gram_schmidt(plain_vectors)

        self.position = nearest[0]
        self.normal = gram_schmidt_vecs[-1]

        print self.position, self.normal

    def predict(self, x):
        dec = dot((x - self.position), self.normal)

        if dec <= 0:
            return False
        else:
            return True

