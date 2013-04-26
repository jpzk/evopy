'''
This file is part of evopy.

Copyright 2012, Jendrik Poloczek

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

Special thanks to Nikolaus Hansen for providing major part of the CMA-ES code.
The CMA-ES algorithm is provided in many other languages and advanced versions at
http://www.lri.fr/~hansen/cmaesintro.html.
'''

from copy import deepcopy

from numpy import array, mean, log, eye, diag, transpose
from numpy import identity, matrix, dot, exp, zeros, ones, sqrt
from numpy.random import normal, rand
from numpy.linalg import eigh, norm

from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from evolution_strategy import EvolutionStrategy

class CMAES(EvolutionStrategy):

    description =\
        "Covariance matrix adaption evolution strategy (CMA-ES)"

    description_short = "CMA-ES"

    def __init__(self, mu, lambd, xmean, sigma):

        # initialize super constructor
        super(CMAES, self).__init__(mu, lambd)

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

        self.ask_pending_solutions

        # dimension of objective function
        N = xmean.size
        self._xmean = xmean
        self._sigma = sigma

        # recombination weights
        self._weights = [log(self._mu + 0.5) - log(i + 1) for i in range(self._mu)]

        # normalize recombination weights array
        self._weights = [w / sum(self._weights) for w in self._weights]

        # variance-effectiveness of sum w_i x_i
        self._mueff = sum(self._weights) ** 2 / sum(w ** 2 for w in self._weights)

        # time constant for cumulation for C
        self._cc = (4 + self._mueff / N) / (N + 4 + 2 * self._mueff / N)

        # t-const for cumulation for sigma control
        self._cs = (self._mueff + 2) / (N + self._mueff + 5)

        # learning rate for rank-one update of C
        self._c1 = 2 / ((N + 1.3) ** 2 + self._mueff)

        # and for rank-mu update
        term_a = 1 - self._c1
        term_b = 2 * (self._mueff - 2 + 1 / self._mueff) / ((N + 2) ** 2 + self._mueff)
        self._cmu = min(term_a, term_b)

        # damping for sigma, usually close to 1
        self._damps = 2 * self._mueff / self._lambd + 0.3 + self._cs

        # evolution paths for C and sigma
        self._pc = zeros(N)
        self._ps = zeros(N)

        # B-matrix of eigenvectors, defines the coordinate system
        self._B = identity(N)

        # diagonal matrix of eigenvalues (sigmas of axes)
        self._D = ones(N)  # diagonal D defines the scaling

        # covariance matrix, rotation of mutation ellipsoid
        self._C = identity(N)
        self._invsqrtC = identity(N)  # C^-1/2

        # approx. norm of random vector
        self._norm = sqrt(N) * (1.0 - (1.0/(4*N)) + (1.0/21*(N**2)))

        ### FIRST RUN
        self._D, self._B = eigh(self._C)
        self._B = matrix(self._B)
        self._D = [d ** 0.5 for d in self._D]

        invD = diag([1.0/d for d in self._D])
        self._invsqrtC = self._B * invD * transpose(self._B)

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for\
            true feasibility """

        pending_solutions = []
        while(len(pending_solutions) < (self._lambd - len(self._valid_solutions))):
            normals = transpose(matrix([normal(0.0, d) for d in self._D]))
            value = self._xmean + transpose(self._sigma * self._B * normals)
            pending_solutions.append(value)

        return pending_solutions

    def tell_feasibility(self, feasibility_information):
        """ tell feasibilty; return True if there are no pending solutions,
            otherwise False """

        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
            else:
                self._count_constraint_infeasibles += 1

        # @todo shorten: return expression
        if(len(self._valid_solutions) < self._lambd):
            return False
        else:
            return True

    def ask_valid_solutions(self):
        return self._valid_solutions

    def tell_fitness(self, fitnesses):
        """ tell fitness; update all strategy specific attributes """

        N = self._xmean.size
        oldxmean = deepcopy(self._xmean)

        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        sorted_fitnesses = sorted(fitnesses, key = fitness)[:self._mu]
        sorted_children = map(child, sorted_fitnesses)

        # new xmean
        values = sorted_children

        self._xmean = matrix([[0.0 for i in range(0,N)]])
        weighted_values = zip(self._weights, values)
        for weight, value in weighted_values:
            self._xmean += weight * value

        # cumulation: update evolution paths
        y = self._xmean - oldxmean
        z = dot(self._invsqrtC, y.T) # C**(-1/2) * (xnew - xold)

        # normalizing coefficient c and evolution path sigma control
        c = (self._cs * (2 - self._cs) * self._mueff) ** 0.5 / self._sigma
        self._ps = (1 - self._cs) * self._ps + c * z

        # normalizing coefficient c and evolution path for rank-one-update
        # without hsig (!)
        c = (self._cc * (2 - self._cc) * self._mueff) ** 0.5 / self._sigma
        self._pc = (1 - self._cc) * self._pc + c * y

        # adapt covariance matrix C
        # rank one update term
        term_cov1 = self._c1 * (transpose(matrix(self._pc)) * matrix(self._pc))

        # ranke mu update term
        valuesv = [(value - oldxmean) / self._sigma for value in values]
        term_covmu = self._cmu *\
            sum([self._weights[i] * (transpose(matrix(valuesv[i])) *\
            matrix(valuesv[i]))\
            for i in range(0, self._mu)])

        self._C = (1 - self._c1 - self._cmu) * self._C + term_cov1 + term_covmu

        # update global sigma by comparing evolution path
        # with approx. norm of random vector
        self._sigma *= exp(self._cs / self._damps) *\
            ((norm(self._ps.getA1()) / self._norm) - 1)

        ### UPDATE FOR NEXT ITERATION
        self._valid_solutions = []

        ### STATISTICS
        self._selected_children = values
        self._best_child, self._best_fitness = sorted_fitnesses[0]
        self._worst_child, self._worst_fitness = sorted_fitnesses[-1]

        fitnesses = map(fitness, sorted_fitnesses)
        self._mean_fitness = array(fitnesses).mean()

        # log all bindings
        self.logger.log()
        self._count_constraint_infeasibles = 0

        self._D, self._B = eigh(self._C)
        self._B = matrix(self._B)
        self._D = [d ** 0.5 for d in self._D]

        invD = diag([1.0/d for d in self._D])
        self._invsqrtC = self._B * invD * transpose(self._B)

        return self._best_child, self._best_fitness

