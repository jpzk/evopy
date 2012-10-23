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
'''
from numpy import array, random, matrix, exp, vectorize
from numpy.random import normal

from evolution_strategy import EvolutionStrategy

class ORIDSES(EvolutionStrategy):

    description =\
        "Ori. Death Penalty Step Control Evolution Strategy (DSES)"    

    description_short = "Ori. DSES" 

    def __init__(self, mu, lambd, theta, pi, initial_sigma,\
        delta, tau0, tau1, initial_pos):

        super(ORIDSES, self).__init__(mu, lambd)

        self._theta = theta
        self._pi = pi
        self._delta = delta
        self._infeasibles = 0
        self._init_pos = initial_pos
        self._init_sigma = initial_sigma
        self._tau0 = tau0
        self._tau1 = tau1

        self._current_population = [] 
        self._valid_solutions = [] 

        self.logger.add_const_binding('_theta', 'theta')
        self.logger.add_const_binding('_pi', 'pi')
        self.logger.add_const_binding('_tau0', 'tau0')
        self.logger.add_const_binding('_tau1', 'tau1')
        self.logger.add_binding('_delta', 'delta')

        # prepare operators, numpy.vectorize for use with matrices
        reducer = lambda sigma : self._delta if sigma < self._delta else sigma
        mutate_pos = lambda coord, sigma : coord + normal(0, sigma)        
        mutate_sig = lambda sigma : sigma * exp(self._tau0 * normal(0,1)) *\
            exp(self._tau1 * normal(0, 1))    

        self._mat_reducer = vectorize(reducer)
        self._mat_mutate_pos = vectorize(mutate_pos)
        self._mat_mutate_sig = vectorize(mutate_sig)

        # log constants
        self.logger.const_log()
        
        # initialize population
        self._initialize_population()                    

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
        child[1] = self._mat_mutate_sig(child[1])

        if(self._infeasibles % self._pi == 0):
            self._delta *= self._theta
 
        # minimum step size
        child[1] = self._mat_reducer(child[1])

        # mutation of position with new step size
        child[0] = self._mat_mutate_pos(child[0], child[1])
       
        return child

    def ask_pending_solutions(self):
        return [self._generate_individual()]

    def tell_feasibility(self, feasibility_information):
        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
            else:
                self._count_constraint_infeasibles += 1
                self._infeasibles += 1

        if(len(self._valid_solutions) < self._lambd):
            self._infeasibles = 0
            return False
        else:
            return True
                
    def ask_valid_solutions(self):
        return self._valid_solutions

    def tell_fitness(self, fitnesses):  
        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        # selection of survival
        sorted_fitnesses = sorted(fitnesses, key = fitness)[:self._mu]
        sorted_children = map(child, sorted_fitnesses)

        """ update the selection probabilites according to 
            anti-proportional fitness. """      
        probabilities = []
        s, a_prop_sum, sum_of_fitnesses = 0.0, 0.0, 0.0

        for individual, fitness in sorted_fitnesses:
            sum_of_fitnesses += fitness
        for individual, fitness in sorted_fitnesses:
            a_prop_sum += 1.0 / (fitness / float(sum_of_fitnesses))
        for individual, fitness in sorted_fitnesses: 
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

        # log information
        self._selected_children = sorted_children
        self._best_child, self._best_fitness = sorted_fitnesses[0]
        self._worst_child, self._worst_fitness = sorted_fitnesses[-1]        
        self._mean_fitness = array(map(lambda (c,f) : f, sorted_fitnesses)).mean()
        
        self.logger.log() 

        # reset generation variables
        self._count_constraint_infeasibles = 0
        self._valid_solutions = [] 
        self._infeasibles = 0

        return self._best_child, self._best_fitness 
