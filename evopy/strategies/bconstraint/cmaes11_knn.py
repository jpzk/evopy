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

Based originally on the paper "A (1+1)-CMA-ES for Constrained Optimisation" by
Arnold, Dirk, V. and Hansen, Nikolaus
'''

from sys import path
path.append("../../..")

from copy import deepcopy

from collections import deque
from numpy import array, mean, log, eye, diag, transpose, arctan2, sin, cos
from numpy import identity, matrix, dot, exp, zeros, ones, sqrt
from numpy.random import normal, rand, random
from numpy.linalg import eigh, norm, inv
from evopy.helper.logger import Logger
from sklearn import linear_model
from evopy.strategies.evolution_strategy import EvolutionStrategy
from sklearn.neighbors import KNeighborsClassifier

class CMAES11KNN(object):

    description =\
        "Covariance matrix adaption evolution strategy (1+1-CMA-ES) with "\
        "binary constraint handling from 'A (1+1)-CMA-ES for Constrained"\
        "Optimisation' with KNN meta model"

    description_short = "1+1-CMA-ES with KNN meta model"

    def __init__(self, xstart, sigma, beta, trainingsize):

        # initialize super constructor
        super(CMAES11KNN, self).__init__()

        self.logger = Logger(self)

        # initialize CMA-ES specific strategy parameters
        self._init_cma_strategy_parameters(xstart, sigma)

        # valid solutions
        self._valid_solutions = []

        self._beta = beta
        self._size = trainingsize
        self._metamodel = KNeighborsClassifier(n_neighbors=self._size)

        # statistics
        self.logger.add_const_binding('_sigma', 'initial_sigma')

        self.logger.add_binding('_A', 'A')
        self.logger.add_binding('_cvec', 'cvec')
        self.logger.add_binding('_infeasible', 'infeasible')
        self.logger.add_binding('_feasible', 'feasible')
        self.logger.add_binding('_feasible_child', 'feasiblechild')
        self.logger.add_binding('_s', 'succpath')
        self.logger.add_binding('_trainingset_feasible', 'tfeasible')
        self.logger.add_binding('_trainingset_infeasible', 'tinfeasible')
        self.logger.add_binding('_feasiblemodel', 'feasiblemodel')
        self.logger.add_binding('_infeasiblemodel', 'infeasiblemodel')

        self._last_best_children = deque(maxlen=20)
        self._last_inf_children = deque(maxlen=20)

        self._trainingset_feasible = []
        self._trainingset_infeasible = []

        N = xstart.size
        self.meta_model_trained = False
        self._last_infeasible = matrix(zeros(N)).T

        self._feasiblemodel = linear_model.LinearRegression()
        self._feasiblemodel.coef_ = 0.0
        self._feasiblemodel.intercept_ = 0.0

        self._infeasiblemodel = linear_model.LinearRegression()
        self._infeasiblemodel.coef_ = 0.0
        self._infeasiblemodel.intercept_ = 0.0

        self._phi = 0
        self._R = matrix(eye(N))

        self._infeasible = []
        self._feasible = []

        # log constants
        self.logger.const_log()

    def _init_cma_strategy_parameters(self, xstart, sigma):

        # dimension of objective function
        N = xstart.size
        self._sigma = sigma

        # damping parameter
        self._n = N
        n = float(N)
        self._d = 1.0 + (n/2.0)
        self._c = 2.0 / (n + 2)
        self._cp = 1.0 / 12.0
        self._ptarget = 2.0 / 11.0
        self._ccovp = 2.0 / (n ** 2 + 6)
        self._cc = 1.0 / (n + 2.0)
        self._beta = 0.1 / (n + 2.0)
        self._psucc = 1.0
        self._s = matrix([1.0 for i in range(0, self._n)]).T

        # covariance matrix, rotation of mutation ellipsoid
        self._A = matrix(identity(N))
        self._cvec = matrix(zeros(N)).T
        self._best_known = False
        self._best_child = xstart
        self._last_best = deque(maxlen=5)

    def _generate_individual(self):
        normals = transpose(matrix([normal(0.0, 1.0) for i in range(0, self._n)]))
        self._z = normals
        value = self._best_child + (self._sigma * self._A * normals).T
        return value

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for\
            true feasibility """

        # use meta model for pre-selection
        individuals = []
        while(len(individuals) < 1):
            if((random() < self._beta) and self.meta_model_trained):
                individual = self._generate_individual()
                if(self._check_feasibility(individual)):
                    individuals.append(individual)
            else:
                individual = self._generate_individual()
                individuals.append(individual)

        return individuals

    def _check_feasibility(self, individual):
        project = lambda x : ((self._R.T * x.T).T).getA1()[0]
        prediction = self._metamodel.predict(project(individual))
        if(prediction == 1):
            return True
        else:
            return False

    def _update_trainingset(self):
        # update rotation matrix && update trainingssets && update
        # metamodel
        m = float(self._feasiblemodel.coef_ + self._infeasiblemodel.coef_)/2.0
        self._phi = arctan2(-1,m)
        phi = self._phi
        self._R = matrix([[cos(phi), -sin(phi)],[sin(phi), cos(phi)]])
        project = lambda x : ((self._R.T * x.T).T).getA1()[0]
        self._trainingset_feasible = map(project, self._last_best_children)
        self._trainingset_infeasible = map(project, self._last_inf_children)

    def _train_trainingset(self):
        lf = len(self._trainingset_feasible)
        li = len(self._trainingset_infeasible)
        if(lf == self._size and li == self._size):
            X = array(self._trainingset_feasible + self._trainingset_infeasible)
            X = X.reshape((X.shape[0],1))
            Y = [1 for i in range(0, self._size)] + [0 for i in range(0, self._size)]
            self._metamodel.fit(X,Y)
            self.meta_model_trained = True
        return

    def tell_feasibility(self, feasibility_information):
        for (child, feasibility) in feasibility_information:
            if(feasibility and (norm(self._z) **2) > 0.5): ### mistake in paper?
                self._feasible_child = child
                self._feasible.append(child)

                if(len(self._last_best_children) >= 1):
                    # estimate linear regression feasible plane
                    X = map(lambda e : e.getA1()[0], self._last_best_children)
                    Y = map(lambda e : e.getA1()[1], self._last_best_children)

                    X = matrix(X).T
                    Y = matrix(Y).T

                    # calculate linear feasiblity model
                    self._feasiblemodel = linear_model.LinearRegression()
                    self._feasiblemodel.fit(X,Y)
                    self._update_trainingset()
                    self._train_trainingset()

                return True
            elif(not feasibility):
                self._last_inf_children.append(child)

                # estimate linear regression infeasible plane
                X = map(lambda e : e.getA1()[0], self._last_inf_children)
                Y = map(lambda e : e.getA1()[1], self._last_inf_children)

                X = matrix(X).T
                Y = matrix(Y).T

                self._infeasiblemodel = linear_model.LinearRegression()
                self._infeasiblemodel.fit(X,Y)
                self._update_trainingset()
                self._train_trainingset()

                self._infeasible.append(child)
                self._last_infeasible = child
                self._cvec = (1 - self._cc) * self._cvec + self._cc * (self._A * self._z)
                wj = (inv(self._A) * self._cvec)
                self._A = self._A - (0.1/4.0) * ((self._cvec * wj.T) / (wj.T * wj))
                return False
            else:
                # infeasible solution update constraint vectors
                return False

    def ask_valid_solutions(self):
        return [self._feasible_child]

    def tell_fitness(self, fitnesses):
        N = self._best_child.size

        child, fitness = fitnesses[0]

        # check if fitness is inferior to last 5 ancestors
        better = True
        for last_fit in self._last_best:
            if(fitness > last_fit):
                better = False

        if better:
            # success in regard to fitness
            A_inv = inv(self._A)
            term_sq = sqrt(1 - self._ccovp)
            term_a = term_sq * self._A
            term_norm = norm(A_inv * self._s) ** 2
            term_fac = term_sq / term_norm
            term_b = sqrt(1 + ((self._ccovp * term_norm) / (1 - self._ccovp))) - 1
            term_c = self._s * (A_inv * self._s).T
            self._A = term_a + term_fac * term_b * term_c

        else:
            term_norm = norm(self._z) ** 2
            self._ccovn = min((0.4/(float(N) ** 1.6 + 1)), (1.0/(2*term_norm-1)))
            term_sq = sqrt(1 + self._ccovn)
            term_a = term_sq * self._A
            term_fac = term_sq / term_norm
            term_b = sqrt(1 - ((self._ccovn * term_norm)/(1 + self._ccovn))) - 1
            term_c = self._A * self._z * self._z.T
            self._A = term_a + term_fac * term_b * term_c

        if self._best_known and fitness <= self._best_fitness:
            self._best_fitness = fitness
            self._best_child = child
            self._last_best.append(fitness)
            self._last_best_children.append(child)

            # update search path
            self._s = (1 - self._c) * self._s +\
                sqrt(self._c * (2 - self._c)) * self._A * self._z
            # update success prob
            self._psucc = (1.0 - self._cp) * self._psucc + self._cp * 1

        elif self._best_known and fitness > self._best_fitness:
            self._psucc = (1.0 - self._cp) * self._psucc + self._cp * 0

        elif not self._best_known:
            self._best_child = child
            self._best_fitness = fitness
            self._best_known = True
            self._s = (1 - self._c) * self._s +\
                sqrt(self._c * (2 - self._c)) * self._A * self._z
            self._last_best.append(fitness)

        # update sigma with success probability
        term_frac = (self._psucc - self._ptarget) / (1.0 - self._ptarget)
        self._sigma = self._sigma * exp((1.0/self._d) * term_frac)

        self.logger.log()
        self._infeasible = []
        self._feasible = []

        return self._best_child, self._best_fitness

