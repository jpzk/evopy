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
from math import floor

from numpy import array, mean, log, eye, diag, transpose
from numpy import identity, matrix, dot, exp, zeros, ones
from numpy.random import normal, rand
from numpy.linalg import eigh, norm, inv

from evolution_strategy import EvolutionStrategy
from evopy.individuals.individual import Individual

class CMAESSVCR(EvolutionStrategy):
 
    description =\
        "Covariance matrix adaption evolution strategy (CMA-ES) with linear SVC "\
        "meta model and repair of infeasibles and mutation ellipsoid alignment"

    description_short = "CMA-ES with SVC"        

    def __init__(self, mu, lambd, xmean, sigma, beta, meta_model):

        # call super constructor 
        super(CMAESSVCR, self).__init__(mu, lambd)

        # initialize CMA-ES specific strategy parameters
        self._init_cma_strategy_parameters(xmean, sigma)      

        # statistics
        self._statistics_constraint_infeasibles_trajectory = []
        self._statistics_repaired_trajectory = []        
        self._count_constraint_infeasibles = 0
        self._count_repaired = 0

        # SVC Metamodel
        self._meta_model = meta_model
        self._meta_model_trained = False
        self._beta = beta

        # valid solutions
        self._valid_solutions = []

    def _init_cma_strategy_parameters(self, xmean, sigma):
        # dimension of objective function
        N = len(xmean) 
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

        ### FIRST RUN
        self._D, self._B = eigh(self._C)
        self._B = matrix(self._B)
        self._D = [d ** 0.5 for d in self._D] 

        invD = diag([1.0/d for d in self._D])
        self._invsqrtC = self._B * invD * transpose(self._B) 

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for true 
            feasibility """

        # testing beta percent of generated children on meta model first.
        pending_meta_feasible = []
        pending_solutions = []

        difference = self._lambd - len(self._valid_solutions)

        if(self._meta_model_trained):
            max_amount_meta_feasible = int(floor(self._beta * difference))
            max_amount_pending_solutions = difference - max_amount_meta_feasible        

            while(len(pending_meta_feasible) < max_amount_meta_feasible):
                normals = transpose(matrix([normal(0.0, d) for d in self._D]))
                value = self._xmean + transpose(self._sigma * self._B * normals)
                individual = Individual(value.getA1()) 

                if(self._meta_model.check_feasibility(individual)):
                    pending_meta_feasible.append(individual)
                else:                    
                    repaired = self._meta_model.repair(individual)
                    self._count_repaired += 1
                    pending_meta_feasible.append(repaired)
        else: 
            max_amount_pending_solutions = difference

        while(len(pending_solutions) < max_amount_pending_solutions):
            normals = transpose(matrix([normal(0.0, d) for d in self._D]))
            value = self._xmean + transpose(self._sigma * self._B * normals)
            pending_solutions.append(Individual(value.getA1()))

        return pending_meta_feasible + pending_solutions            

    def tell_feasibility(self, feasibility_information):
        """ tell feasibilty; return True if there are no pending solutions, 
            otherwise False """

        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
            else:
                self._count_constraint_infeasibles += 1
                self._meta_model.add_infeasible(child)

        if(len(self._valid_solutions) < self._lambd):
            return False
        else:            
           return True

    def ask_valid_solutions(self):
        return self._valid_solutions

    def tell_fitness(self, fitnesses):
        """ tell fitness; update all strategy specific attributes """       

        N = len(self._xmean)
        oldxmean = deepcopy(self._xmean)

        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        sorted_fitnesses = sorted(fitnesses, key = fitness)
        sorted_children = map(child, sorted_fitnesses)

        # update meta model sort self._valid_solutions by fitness and 
        # unsorted self._sliding_infeasibles
        self._meta_model.add_sorted_feasibles(sorted_children)       
        self._meta_model_trained = self._meta_model.train()

        # new xmean
        values = map(lambda child : child.value, sorted_children[:self._mu]) 
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
            sum([self._weights[i] * (transpose(matrix(valuesv[i])) *\
            matrix(valuesv[i]))\
            for i in range(0, self._mu)])

        self._C = (1 - self._c1 - self._cmu) * self._C + term_cov1 + term_covmu

        #update sigma page. 20, equation (30)
        self._sigma *= exp(min(0.6, (self._cs / self._damps) *\
            sum(x ** 2 for x in self._ps.getA1())/(N - 1) / 2))

        ### UPDATE FOR NEXT ITERATION
        self._valid_solutions = []

        ### STATISTICS
        self._statistics_constraint_infeasibles_trajectory.append(\
            self._count_constraint_infeasibles)        
        self._count_constraint_infeasibles = 0                

        self._statistics_repaired_trajectory.append(\
            self._count_repaired)
        self._count_repaired = 0            

        self._statistics_selected_children_trajectory.append(values)

        # update best child, best fitness
        best_child, best_fitness = sorted_fitnesses[0]
        worst_child, worst_fitness = sorted_fitnesses[-1]        

        fitnesses = map(fitness, sorted_fitnesses)
        mean_fitness = array(fitnesses).mean()

        self._statistics_best_fitness_trajectory.append(best_fitness)
        self._statistics_worst_fitness_trajectory.append(worst_fitness)
        self._statistics_mean_fitness_trajectory.append(mean_fitness)

        self._D, self._B = eigh(self._C)
        self._B = matrix(self._B)
        self._D = [d ** 0.5 for d in self._D] 

        invD = diag([1.0/d for d in self._D])
        self._invsqrtC = self._B * invD * transpose(self._B) 

        return best_child, best_fitness

    def _blend_B_with_rotation(self, B, rotation):
    
        blend_pairs = []
        taken = [False for i in range(0, len(rotation.T))]

        simialarity = 0.0

        for i in range(0, len(B.T)):
            first = True
            vector_in_b = B.T[i]
            for j in range(0, len(rotation.T)):
                vector_in_rot = rotation.T[j].T
                product = dot(vector_in_b.getA1(), vector_in_rot.getA1())
                if(not taken[j] and first): 
                        first = False
                        closest = vector_in_rot            
                        closest_product = product
                        closest_index = j
                else:
                    if(not taken[j] and product > closest_product):
                        closest = vector_in_rot
                        closest_product = product
                        closest_index = j
            simialarity += closest_product                        
            blend_pairs.append((vector_in_b, closest))
            taken[j] = True

        simialarity /= 2.0
        simialarity = max(0.0, simialarity)
        print "sim", simialarity
        blended_mat = []

        #blend_factor = 1.0 - simialarity
        blend_factor = 0.0
        print "blendfactor", blend_factor
 
        for (b_vec, rot_vec) in blend_pairs:
            blended_vec = (1.0 - blend_factor) * b_vec + blend_factor * rot_vec.getA1()
            blended_mat.append(blended_vec.getA1())

        return matrix(blended_mat).T

    def get_statistics(self):
        statistics = {
            "infeasibles" : self._statistics_constraint_infeasibles_trajectory,
            "repaired": self._statistics_repaired_trajectory}
       
        super_statistics = super(CMAESSVCR, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]

        return statistics

    def get_last_statistics(self):
        statistics = {
            "infeasibles" : self._statistics_constraint_infeasibles_trajectory[-1],
            "repaired": self._statistics_repaired_trajectory[-1]}
 
        super_statistics = super(CMAESSVCR, self).get_last_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]

        return statistics
