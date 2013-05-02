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
from numpy.linalg import eigh, norm

from evopy.strategies.evolution_strategy import EvolutionStrategy

class CMAES11(EvolutionStrategy):

    description =\
        "Covariance matrix adaption evolution strategy (1+1-CMA-ES) with "\
        "binary constraint handling from A (1+1)-CMA-ES for Constrained"\
        "Optimisation"

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

        self.logger.add_binding('_D', 'D')
        self.logger.add_binding('_C', 'C')
        self.logger.add_binding('_B', 'B')

        # log constants
        self.logger.const_log()

    def _init_cma_strategy_parameters(self, xmean, sigma):

        # dimension of objective function
        N = xmean.size
        self._xmean = xmean
        self._sigma = sigma

        # damping parameter
        n = float(N)
        self._d = 1.0 + (n/2.0)
        self._c = 2.0 / (n + 2)
        self._cp = 1.0 / 12.0
        self._ptarget = 2.0 / 11.0
        self._ccovp = 2.0 / (n ** 2 + 6)
        self._cc = 1.0 / (n + 2.0)
        self._beta = 0.1 / (n + 2.0)
        self._psucc = 1.0
        self._s = zeros(N)

        # covariance matrix, rotation of mutation ellipsoid
        self._C = identity(N)

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for\
            true feasibility """

        pending_solutions = []
        while(len(pending_solutions) < (self._lambd - len(self._valid_solutions))):
            normals = transpose(matrix([normal(0.0, d) for d in self._D]))
            value = self._xmean + transpose(self._sigma * self._B * normals)
            pending_solutions.append(value)

        return pending_solutions
