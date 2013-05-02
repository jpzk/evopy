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

from sys import path
path.append("../../..")

from copy import deepcopy

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

    def __init__(self, mu, lambd, xmean, sigma):

        # initialize super constructor
        super(CMAES11, self).__init__(mu, lambd)

        # initialize CMA-ES specific strategy parameters
        self._init_cma_strategy_parameters(xmean, sigma)

        # valid solutions
        self._valid_solutions = []

        # statistics
        self.logger.add_const_binding('_xmean', 'initial_xmean')
        self.logger.add_const_binding('_sigma', 'initial_sigma')

        self.logger.add_binding('_A', 'A')

        # log constants
        self.logger.const_log()

    def _init_cma_strategy_parameters(self, xmean, sigma):

        # dimension of objective function
        N = xmean.size
        self._xmean = xmean
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
        self._s = matrix(zeros(N))

        # covariance matrix, rotation of mutation ellipsoid
        self._A = matrix(identity(N))
        self._cvec = matrix(zeros(N))

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for\
            true feasibility """

        normals = transpose(matrix([normal(0.0, 1.0) for i in range(0, self._n)]))
        value = self._xmean + transpose(self._sigma * self._A * normals)

        return [value]

    def tell_feasibility(self, feasibility_information):
        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._feasible_child = child
                return True
            else:
                # infeasible solution update constraint vectors
                self._cvec = (1 - self._cc) * self._cvec + self._cc * child
                return False

    def ask_valid_solutions(self):
        return [self._feasible_child]

    def tell_fitness(self, fitnesses):
        N = self._xmean.size
        oldxmean = deepcopy(self._xmean)

        # success in regard to fitness
        # update A
        A_inv = inv(self._A)
        term_sq = sqrt(1 - self._ccovp)
        term_a = term_sq * self._A
        term_norm = norm(A_inv * self._s.T)
        term_fac = term_sq / term_norm
        term_b = sqrt(1 + (self._ccovp * term_norm) / (1 - self._ccovp) - 1)
        term_c = self._s.T * self._s * A_inv.T
        self._A = term_sq * self._A + term_fac * term_b + term_c

        child, fitness = fitnesses[0]

        # unsuccess in regard to fitness

        # update sigma with success probability
        term_frac = (self._psucc - self._ptarget) / (1.0 - self._ptarget)
        self._sigma = self._sigma * exp((1.0/self.d)) * term_frac
