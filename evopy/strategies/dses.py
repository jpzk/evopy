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

from math import sqrt
from numpy import array, random, matrix, exp, vectorize

from evolution_strategy import EvolutionStrategy

class DSES(EvolutionStrategy):

    description =\
        "Death Penalty Step Control Evolution Strategy (DS-ES)"    

    description_short = "DS-ES"        

    def __init__(self, mu, lambd, theta, pi, initial_sigma,\
        delta, tau0, tau1, initial_pos):

        super(DSES, self).__init__(mu, lambd)

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

        # log constants
        self.logger.const_log()
        
        # initialize population
        self._initialize_population()                    

    def _initialize_population(self):
        init_pos, init_sigma = self._init_pos, self._init_sigma
        d = len(init_pos)

        genpos = lambda pos, sigma : random.normal(pos, sigma)
        gensig = lambda sigma : sigma 
         
        while(len(self._current_population) < self._lambd):
            sigmas = [gensig(init_sigma[i]) for i in range(0, d)]
            positions = [genpos(init_pos[i], sigmas[i]) for i in range(0, d)]
            individual = matrix([positions, sigmas])
            self._current_population.append(individual)

    def _generate_individual(self):
        # recombination
        e1 = self._current_population[random.randint(0, self._mu)]
        e2 = self._current_population[random.randint(0, self._mu)]
        child = 0.5 * (e1 + e2)

        # mutation of sigma
        normal = random.normal
        temp = exp(self._tau0 * normal(0, 1))
        mutate = lambda sigma : temp * exp(self._tau1 * normal(0, sigma))
        child[1] = vectorize(mutate)(child[1])

        # minimum step size
        delta = self._delta
        reducer = lambda sigma : delta if sigma < delta else sigma        
        child[1] = vectorize(reducer)(child[1])

        # mutation of position with new step size
        mutate = lambda coord, sigma : coord + normal(0, sigma)        
        child[0] = vectorize(mutate)(child[0], child[1])
       
        return child

    def ask_pending_solutions(self):
        pending_solutions = []

        # death penalty
        while(len(pending_solutions) < (self._lambd - len(self._valid_solutions))):
            if(self._infeasibles > self._pi):
                self._delta *= self._theta
                self._infeasibles = 0 # not in original algorithm
            child = self._generate_individual()
            pending_solutions.append(child)

        return pending_solutions

    def tell_feasibility(self, feasibility_information):
        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
            else:
                self._count_constraint_infeasibles += 1
                self._infeasibles += 1

        if(len(self._valid_solutions) < self._lambd):
            return False
        else:
            return True
                
    def ask_valid_solutions(self):
        return self._valid_solutions

    def tell_fitness(self, fitnesses):   
        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        # selection
        sorted_fitnesses = sorted(fitnesses, key = fitness)[:self._mu]
        sorted_children = map(child, sorted_fitnesses)
        self._current_population = sorted_children

        # log information
        self._selected_children = array(sorted_children)
        self._best_child, self._best_fitness = sorted_fitnesses[0]
        self._worst_child, self._worst_fitness = sorted_fitnesses[-1]        
        self.logger.log() 

        # reset generation variables
        self._count_constraint_infeasibles = 0
        self._valid_solutions = [] 
        self._infeasibles = 0

        print self._best_child[1]

        return self._best_child, self._best_fitness 
