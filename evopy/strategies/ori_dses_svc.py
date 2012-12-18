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

from copy import deepcopy
from math import floor
from numpy import array, random, matrix, exp, vectorize
from numpy.random import normal

from evolution_strategy import EvolutionStrategy
from confusion_matrix import ConfusionMatrix

# constants, row indices for individual matrix
POS = 0
SIGMA = 1

class ORIDSESSVC(EvolutionStrategy):

    description =\
        "Ori. Death Penalty Step Control Evolution Strategy (DSES) with SVC"

    description_short = "Ori. DSES with SVC"

    def __init__(self, mu, lambd, theta, pi, initial_sigma,\
        delta, tau0, tau1, initial_pos, beta, meta_model):

        super(ORIDSESSVC, self).__init__(mu, lambd)

        self._theta = theta
        self._pi = pi
        self._delta = delta
        self._infeasibles = 0
        self._init_pos = initial_pos
        self._init_sigma = initial_sigma
        self._tau0 = tau0
        self._tau1 = tau1

        # SVC Metamodel
        self.meta_model = meta_model
        self.meta_model_trained = False
        self._beta = beta

        self._current_population = [] 
        self._valid_solutions = [] 
        self._pending_apos_solutions = []

        self.logger.add_const_binding('_theta', 'theta')
        self.logger.add_const_binding('_pi', 'pi')
        self.logger.add_const_binding('_tau0', 'tau0')
        self.logger.add_const_binding('_tau1', 'tau1')
        self.logger.add_binding('_delta', 'delta')
        self.logger.add_binding('_sp', 'successprob')
        self.logger.add_binding('_ppv', 'ppv')
        self.logger.add_binding('_npv', 'npv')

        # log constants
        self.logger.const_log()
       
        # initialize population 
        self._initialize_population()

    # cPickle cannot serialize lambda functions 
    def _mat_mutate_sig(self, sig):
        mutate_sig = lambda sigma : sigma * exp(self._tau1 * normal(0, 1)) 
        _lmutatesig = vectorize(mutate_sig)
        return _lmutatesig(sig)

    # cPickle cannot serialize lambda functions 
    def _mat_mutate_pos(self, coord, sigma):
        mutate_pos = lambda coord, sigma : coord + normal(0, sigma) 
        _lmutatepos = vectorize(mutate_pos)
        return _lmutatepos(coord, sigma)

    # cPickle cannot serialize lambda functions 
    def _mat_reducer(self, x):
        reducer = lambda sigma : self._delta if sigma < self._delta else sigma
        _lmatreducer = vectorize(reducer)
        return _lmatreducer(x)

    def _initialize_population(self):
        init_pos, init_sigma = self._init_pos, self._init_sigma
        d = init_pos.size

        genpos = lambda pos, sigma : random.normal(pos, sigma)
        gensig = lambda sigma : sigma 
         
        # initial mu lambda population, with selection of pairing 
        # probability 1/mu. interval size is equally.
        s, i = 0.0, (1 / float(self._mu))
        while(len(self._current_population) < self._mu):
            sigma = self._mat_mutate_sig(init_sigma)
            pos = self._mat_mutate_pos(init_pos, sigma)
            individual = matrix([pos.getA1(), sigma.getA1()])  
            self._current_population.append((individual, s, s+i))
            s = s+i        

    def _generate_individual(self):
        # selection of pairing, anti-proportional selection using 
        # the intervals between [0, 1]
        parents = []
        while(len(parents) < 2): 
            x = random.random()
            for individual, start, end in self._current_population:
                if(start <= x < end):
                    parents.append(individual) 

        child = 0.5 * (parents[0] + parents[1])

        # mutation of sigma
        self._global_sigma_mutation = exp(self._tau0 * normal(0, 1))
        child[SIGMA] = self._mat_mutate_sig(child[SIGMA])
        child[SIGMA] = self._global_sigma_mutation * child[SIGMA]

        if(self._infeasibles % self._pi == 0):
            self._delta *= self._theta
 
        # minimum step size
        child[SIGMA] = self._mat_reducer(child[SIGMA])

        # mutation of position with new step size
        child[POS] = self._mat_mutate_pos(child[POS], child[SIGMA])
       
        return child

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for true 
            feasibility """
        
        individuals = []
        while(len(individuals) < 1):
            if((random.random() < self._beta) and self.meta_model_trained):
                individual = self._generate_individual() 
                if(self.meta_model.check_feasibility(individual[POS])):
                    individuals.append(individual)
                    # appending meta-feasible solution to a_posteriori pending
                    self._pending_apos_solutions.append((individual, True))
                else:
                    # appending meta-infeasible solution to a_posteriori pending 
                    self._pending_apos_solutions.append((individual, False))

                #individual[POS] = self.meta_model.repair(individual[POS])
                #self._count_repaired += 1
                #pending_meta_feasible.append(individual)

                # appending meta-feasible solution to a_posteriori pending
                #self._pending_apos_solutions.append((individual, True))
            else: 
                individual = self._generate_individual()
                individuals.append(individual)

        return individuals           
   
    def tell_feasibility(self, feasibility_information):
        """ tell feasibilty; return True if there are no pending solutions, 
            otherwise False """

        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
                self._infeasibles = 0
            else:
                self._count_constraint_infeasibles += 1
                self._infeasibles += self._infeasibles
                self.meta_model.add_infeasible(child[POS])

        if(len(self._valid_solutions) < self._lambd):
            return False
        else:            
           return True

    def ask_valid_solutions(self):
        return self._valid_solutions

    def ask_a_posteriori_solutions(self):
        return self._pending_apos_solutions        

    def tell_fitness(self, fitnesses):
        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child
        position = lambda (child, fitness) : child[POS]

        sorted_fitnesses = sorted(fitnesses, key = fitness)
        sorted_children = map(child, sorted_fitnesses)      
        selected_sorted_fitnesses = sorted_fitnesses[:self._mu]

        # update meta model sort self._valid_solutions by fitness and 
        # unsorted self._sliding_infeasibles
        sorted_feasibles = map(position, sorted_fitnesses)        
        self.meta_model.add_sorted_feasibles(sorted_feasibles)       
        self.meta_model_trained = self.meta_model.train()

        """ update the selection probabilites according to 
            anti-proportional fitness. """      
        probabilities = []
        s, a_prop_sum, sum_of_fitnesses = 0.0, 0.0, 0.0

        for individual, fitness in selected_sorted_fitnesses:
            sum_of_fitnesses += fitness
        for individual, fitness in selected_sorted_fitnesses:
            a_prop_sum += 1.0 / (fitness / float(sum_of_fitnesses))
        for individual, fitness in selected_sorted_fitnesses: 
            p = (1.0 / (fitness / float(sum_of_fitnesses))) / a_prop_sum
            probabilities.append((individual, p))
        probabilities.reverse()
    
        """ update the current population """            
        self._current_population = []
        start = 0
        for individual, prob in probabilities:
            self._current_population.append((individual, start, start + prob))
            start = s + prob
        self._current_population.reverse() 

        ### UPDATE FOR NEXT ITERATION
        self._valid_solutions = []
        
        ### STATISTICS
        self._selected_children = self._current_population  
        self._best_child, self._best_fitness = selected_sorted_fitnesses[0]
        self._worst_child, self._worst_fitness = selected_sorted_fitnesses[-1]
        self._mean_fitness = array(map(lambda (c,f) : f, selected_sorted_fitnesses)).mean()

        return self._best_child, self._best_fitness

    def tell_a_posteriori_feasibility(self, apos_feasibility):
        self._confusion_matrix = ConfusionMatrix(apos_feasibility)
        self._sp = self._confusion_matrix.success_probability()
        self._ppv = self._confusion_matrix.ppv()
        self._npv = self._confusion_matrix.npv()
        self._pending_apos_solutions = []

        # log all bindings
        self.logger.log()
        self._count_constraint_infeasibles = 0                
        self._count_repaired = 0

