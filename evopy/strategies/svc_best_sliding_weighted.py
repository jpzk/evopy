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

from math import floor
from collections import deque # used for sliding window for best
from svc_evolution_strategy import SVCEvolutionStrategy

class SVCBestSlidingWeighted(SVCEvolutionStrategy):
    """ Using the fittest feasible and infeasible individuals in a sliding
        window (between generations) to build a meta model using SVC. """

    def __init__(\
        self, problem, mu, lambd, alpha, sigma, combination, mutation,\
        selection, view, beta, window_size, append_to_window,\
        parameter_C, parameter_gamma):

        super(SVCBestSlidingWeighted, self).__init__(\
            problem, mu, lambd, alpha, sigma, combination, mutation,\
            selection, view, parameter_C, parameter_gamma)

        self._beta = beta 
        self._window_size = window_size
        self._append_to_window = append_to_window
        self._sliding_best_feasibles = deque(maxlen = self._window_size)
        self._sliding_best_infeasibles = deque(maxlen = self._window_size)

    def generate_child(self, population, sigma):
        combined_child = self.combine(population)
        child = self.mutate(combined_child, sigma)
        return child

    def _run(self, (population, generation, m, l, lastfitness,\
        alpha, sigma)):
        """ This method is called every generation. """

        children = [self.generate_child(population, sigma) for child in range(0,l)]

        # Filter by checking feasiblity with SVC meta model, the 
        # meta model might be wrong, so we have to weighten between
        # the filtern with meta model and filtering with the 
        # true constraint function.
        cut = int(floor(self._beta * len(children)))
        meta_children = children[:cut]        
        constraint_children = children[cut:]

        # Filter by meta model
        meta_feasible_children = filter(self.is_meta_feasible, meta_children)
        
        # Filter by true feasibility with constraind function, here we
        # can update the sliding feasibles and infeasibles.
        feasible_children = []
        infeasible_children = []
       
        # Death penalty when feasible by meta model but is infeasible
        for meta_feasible in meta_feasible_children:             
            if(self.is_feasible(meta_feasible)):
                feasible_children.append(meta_feasible)                
            else:
                infeasible_children.append(meta_feasible)
                # Because of Death Penalty we need a feasible reborn.
                reborn = []
                while(len(reborn) < 1):  
                    generated = self.generate_child(population, sigma) 
                    if(self.is_feasible(generated)):
                        reborn.append(generated)
                feasible_children.extend(reborn)                  

        # Filter the other part of the cut with the true constraint function. 
        # Using this information to update the meta model.
        for child in constraint_children:
            if(self.is_feasible(child)):
                feasible_children.append(child)
            else:
                infeasible_children.append(child) 
                # Because of Death Penalty we need a feasible reborn.
                reborn = []
                while(len(reborn) < 1):
                    generated = self.generate_child(population, sigma) 
                    if(self.is_feasible(generated)):
                        reborn.append(generated)
                feasible_children.extend(reborn)

        # feasible_children contains exactly lambda feasible children
        # and infeasible_children 
        next_population = self.select(population + feasible_children, m)
        
        map(self._sliding_best_infeasibles.append,
            self.select(infeasible_children, self._append_to_window))

        map(self._sliding_best_feasibles.append,
            self.select(feasible_children, self._append_to_window))

        self.train_metamodel(\
            feasible = self._sliding_best_feasibles,
            infeasible = self._sliding_best_infeasibles,
            parameter_C = self._parameter_C,
            parameter_gamma = self._parameter_gamma)
      
        fitness_of_best = self.fitness(next_population[0])
        
        self.log(generation, next_population)
        self.view(generation, next_population)

        if(self.termination(generation, fitness_of_best)):
            print next_population[0]
            return True
        else:
            return (next_population, generation + 1, m,\
            l, fitness_of_best, alpha, sigma)

    def run(self):
        """ This method initializes the population etc. And starts the 
            recursion. """
       
        # check for feasiblity and initialize sliding feasible and 
        # infeasible populations.

        feasible_parents = []
        best_feasibles = []
        feasibles = []
        best_infeasibles = []
        infeasibles = []

        # generate a feasible population

        while(len(feasible_parents) < self._mu):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasible_parents.append(parent)
                feasibles.append(parent)
            else:
                infeasibles.append(parent)

        # initial training of the meta model
        # just to be sure 
        while(len(infeasibles) < self._window_size):
            parent = self.generate_population() 
            if(not self.is_feasible(parent)):
                infeasibles.append(parent)

        while(len(feasibles) < self._window_size):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasibles.append(parent)

        best_feasibles = self.select(feasibles, self._window_size)
        best_infeasibles = self.select(infeasibles, self._window_size)

        self.train_metamodel(\
            feasible = best_feasibles,
            infeasible = best_infeasibles,
            parameter_C = self._parameter_C,
            parameter_gamma = self._parameter_gamma)

        result = self._run((feasible_parents, 0, self._mu,\
            self._lambd, 0, self._alpha, self._sigma))

        while result != True:
            result = self._run(result)

        return result

"""
if __name__ == "__main__":
    method = SVCBestSlidingWeighted(\
        SphereProblem(),
        mu = 15,
        lambd = 100,
        alpha = 0.5,
        sigma = 1,
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        parameter_C = 1.0,
        parameter_gamma = 0.0)
     
    method.run()
""" 
