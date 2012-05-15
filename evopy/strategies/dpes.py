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

from evolution_strategy import EvolutionStrategy

class DPES(EvolutionStrategy): 

    def __init__(self, problem, mu, lambd, sigma,\
        combination, mutation, selection, view):

        super(DPES, self).\
            __init__(problem, mu, lambd, \
            combination, mutation, selection, view, sigma = sigma) 

    def _run(self, (population, generation, m, l, lastfitness,\
        alpha, sigma)):

        feasible_children = []
        while(len(feasible_children) < l): 
            combined_child = self.combine(population)
            child = self.mutate(combined_child, sigma)
            
            if(self.is_feasible(child)):
                feasible_children.append(child)
        
        next_population = self.select(population, feasible_children, m)
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
        feasible_parents = []
        while(len(feasible_parents) < self._mu):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasible_parents.append(parent)

        result = self._run((\
            feasible_parents, 0, self._mu, self._lambd, 0, 
            self._alpha, self._sigma))

        while result != True:
            result = self._run(result)

        return result
