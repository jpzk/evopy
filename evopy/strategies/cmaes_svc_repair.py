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
from collections import deque 

from numpy import array, mean, log, eye, diag, transpose
from numpy import identity, matrix, dot, exp, zeros, ones
from numpy.random import normal, rand
from numpy.linalg import eigh, norm

from evopy.individuals.individual import Individual
from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from mm_evolution_strategy import MMEvolutionStrategy

class CMAESSVCR(MMEvolutionStrategy):
    """ Using the fittest feasible and infeasible individuals in a sliding
        window (between generations) to build a meta model using SVC. """
 
    _strategy_name =\
        "Covariance matrix adaption evolution strategy (CMA-ES) with linear SVC "\
        "meta model and repair of infeasibles and mutation ellipsoid alignment"

    def __init__(\
        self, problem, mu, lambd, combination, mutation, selection, xmean, sigma,\
        view, beta, window_size, append_to_window, scaling,\
        crossvalidation, repair_mode = 'mirror'):

        super(CMAESSVCR, self).__init__(\
            problem, mu, lambd, combination, mutation, selection, view)

        # initialize CMA-ES specific strategy parameters
        self._init_cma_strategy_parameters(xmean, sigma)      

        # statistics
        self._statistics_parameter_C_trajectory = []
        self._statistics_DSES_infeasibles_trajectory = []
        self._statistics_angles_trajectory = []
        
        # SVC Metamodel
        self._meta_model = SVCLinearMetaModel() 
        self._beta = beta
        self._append_to_window = append_to_window
        self._window_size = window_size
        self._scaling = scaling            
        self._sliding_best_feasibles = deque(maxlen = self._window_size)
        self._sliding_best_infeasibles = deque(maxlen = self._window_size)
        self._crossvalidation = crossvalidation
        self._repair_mode = repair_mode

    # main evolution 
    def _run(self, (population, generation, m, l, lastfitness)):
        """ This method is called every generation. """

        DSES_infeasibles = 0
        meta_infeasibles = 0

        # ASK part
        # eigendecomposition of C into D and B.
        self._D, self._B = eigh(self._C)
        self._B = matrix(self._B)
        self._D = [d ** 0.5 for d in self._D] 

        invD = diag([1.0/d for d in self._D])
        self._invsqrtC = self._B * invD * transpose(self._B) 

        children = []
        while(len(children) < self._lambd):
            normals = transpose(matrix([normal(0.0, d) for d in self._D]))
            value = self._xmean + transpose(self._sigma * self._B * normals)
            child = Individual(value.getA1())
            if(self.is_feasible(child)):
                children.append(child)
            else:
                DSES_infeasibles += 1
        
        N = len(self._xmean)

        # TELL part
        oldxmean = deepcopy(self._xmean)
        sort_by = lambda child : self.fitness(child)
        sorted_children = sorted(children, key = sort_by)[:self._mu]
         
        self._best_fitness = self.fitness(sorted_children[0])
        self._best_child = deepcopy(sorted_children[0])

        # new xmean
        values = map(lambda child : child.value, sorted_children) 
        self._xmean = dot(self._weights, values)
       
        # cumulation: update evolution paths
        y = self._xmean - oldxmean
        z = dot(self._invsqrtC, y) # C**(-1/2) * (xnew - xold)

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
            sum([self._weights[i] * (transpose(matrix(valuesv[i])) * matrix(valuesv[i]))\
            for i in range(0, self._mu)])

        self._C = (1 - self._c1 - self._cmu) * self._C + term_cov1 + term_covmu

        #update sigma page. 20, equation (30)
        self._sigma *= exp(min(0.6, (self._cs / self._damps) *\
            sum(x ** 2 for x in self._ps.getA1())/(N - 1) / 2))
                            
        best_acc = 0.0
        best_parameter_C = 0.0 
        fitness_of_best = self._best_fitness
        next_population = sorted_children
       
        self.log(generation, next_population, best_acc,\
            best_parameter_C, DSES_infeasibles, meta_infeasibles,\
            [0.0])#self._mutation.get_angles_degree())

        self._view.view(generations = generation,\
            best_fitness = fitness_of_best, best_acc = best_acc,\
            parameter_C = best_parameter_C, DSES_infeasibles = DSES_infeasibles,\
            wrong_meta_infeasibles = meta_infeasibles,\
            angles = [0.0])

        if(self.termination(generation, fitness_of_best)):
            return True
        else:
            return (next_population, generation + 1, m,\
            l, fitness_of_best)

    def run(self):
        """ This method initializes the population etc. And starts the 
            recursion. """
       
        children = []
        while(len(children) < self._lambd):
            normals = transpose(matrix([normal(0.0, d) for d in self._D]))
            value = self._xmean + transpose(self._sigma * self._B * normals)
            child = Individual(value.getA1())
            if(self.is_feasible(child)):
                children.append(child)
 
        result = self._run((children, 0, self._mu, self._lambd, 0))

        while result != True:
            result = self._run(result)

        return result

    def _init_cma_strategy_parameters(self, xmean, sigma):
        # dimension of objective function
        N = self._problem._d
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
 
    def log(\
        self, generation, next_population, best_acc, best_parameter_C,\
        DSES_infeasibles, wrong_meta_infeasibles, angles):
        
        super(CMAESSVCR, self).log(generation, next_population, best_acc,\
            wrong_meta_infeasibles) 

        self._statistics_parameter_C_trajectory.append(best_parameter_C)
        self._statistics_DSES_infeasibles_trajectory.append(DSES_infeasibles)
        self._statistics_angles_trajectory.append(angles)

    def get_statistics(self):
        statistics = {
            "parameter-C" : self._statistics_parameter_C_trajectory,
            "DSES-infeasibles" : self._statistics_DSES_infeasibles_trajectory,
            "angle": self._statistics_angles_trajectory}
        
        super_statistics = super(CMAESSVCR, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]
        
        return statistics

    # return true if solution is feasible in meta model, otherwise false.
    def is_meta_feasible(self, x):
        self._count_is_meta_feasible += 1
        return self._meta_model.check_feasibility(x)

    # train the metamodel with given points
    def train_metamodel(self, feasible, infeasible, parameter_C):
        self._count_train_metamodel += 1
        self._meta_model.train(feasible, infeasible, parameter_C)

    # mutate child with gauss devriation 
    def mutate(self, child, sigmas):
        self._statistics_mutations += 1
        return self._mutation.mutate(child, sigmas)

