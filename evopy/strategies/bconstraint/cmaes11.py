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

Based on the paper "A (1+1)-CMA-ES for Constrained Optimisation" by
Arnold, Dirk, V. and Hansen, Nikolaus
'''

from sys import path
path.append("../../..")

from copy import deepcopy

from collections import deque
from numpy import array, mean, log, eye, diag, transpose
from numpy import identity, matrix, dot, exp, zeros, ones, sqrt
from numpy.random import normal, rand
from numpy.linalg import eigh, norm, inv

from evopy.strategies.evolution_strategy import EvolutionStrategy

class CMAES11(EvolutionStrategy):

    description =\
        "Covariance matrix adaption evolution strategy (1+1-CMA-ES) with "\
        "binary constraint handling from 'A (1+1)-CMA-ES for Constrained"\
        "Optimisation'"

    description_short = "1+1-CMA-ES"

    def __init__(self, mu, lambd, xstart, sigma):

        # initialize super constructor
        super(CMAES11, self).__init__(mu, lambd)

        # initialize CMA-ES specific strategy parameters
        self._init_cma_strategy_parameters(xstart, sigma)

        # valid solutions
        self._valid_solutions = []

        # statistics
        self.logger.add_const_binding('_sigma', 'initial_sigma')

        self.logger.add_binding('_A', 'A')

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

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for\
            true feasibility """

        normals = transpose(matrix([normal(0.0, 1.0) for i in range(0, self._n)]))
        self._z = normals
        value = self._best_child + (self._sigma * self._A * normals).T
        return [value]

    def tell_feasibility(self, feasibility_information):
        for (child, feasibility) in feasibility_information:
            if(feasibility and (norm(self._z) **2) > 0.5): ### mistake in paper?
                self._feasible_child = child
                return True
            else:
                # infeasible solution update constraint vectors
                self._cvec = (1 - self._cc) * self._cvec + self._cc * (self._A * self._z)
                wj = (inv(self._A) * self._cvec)
                self._A = self._A - (0.1/4.0) * ((self._cvec * wj.T) / (wj.T * wj))
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
            # update search path
            self._s = (1 - self._c) * self._s +\
                sqrt(self._c * (2 - self._c)) * self._A * self._z
            # update success prob
            self._psucc = (1.0 - self._cp) * self._psucc + self._cp * 1

        elif self._best_known and fitness > self._best_fitness:
            self._psucc = (1.0 - self._cp) * self._psucc + self._cp * 0
            return self._best_child, self._best_fitness

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

        return self._best_child, self._best_fitness

