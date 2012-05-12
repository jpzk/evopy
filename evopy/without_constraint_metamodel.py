#!/bin/env python
# encoding utf-8

''' 
This file is part of evolutionary-algorithms-sandbox.

evolutionary-algorithms-sandbox is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evolutionary-algorithms-sandbox is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evolutionary-algorithms-sandbox.  If not, see <http://www.gnu.org/licenses/>.
'''

from evolution_strategy import EvolutionStrategy
from sphere_problem import SphereProblem

class WithoutConstraintMetaModel(EvolutionStrategy): 

    def __init__(self, problem, mu, lambd, alpha, sigma):
        super(WithoutConstraintMetaModel, self).\
            __init__(problem, mu, lambd, alpha, sigma)                 

    # main evolution 
    def _run(self, (population, generation, m, l, lastfitness,\
        alpha, sigma)):

        super(WithoutConstraintMetaModel, self)._run(\
            population, generation, m, l, lastfitness, alpha, sigma)

        feasible_children = []
        while(len(feasible_children) < l): 
            child = self.generate_child(population, sigma) 
            if(self.is_feasible(child)):
                feasible_children.append(child)
        
        next_population =\
            self.sortedbest(population + feasible_children)[:m]
  
        fitness_of_best = self.fitness(next_population[0])
        fitness_of_worst = self.fitness(\
            next_population[len(next_population) - 1])

        # only for visual output purpose.
        print "generation " + str(generation) +\
        " smallest fitness " + str(fitness_of_best) 

        new_sigma = sigma
    
        if(2 - 1 * pow(10, -2) < fitness_of_best < 2 + 1 * pow(10, -2)):
            print next_population[0]
            return True
        else:
            return (next_population, generation + 1, m,\
            l, fitness_of_best, alpha, new_sigma)

    def run(self):
        # check for feasiblity and initialize sliding feasible and 
        # infeasible populations.

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

if __name__ == "__main__":
    method = WithoutConstraintMetaModel(\
        SphereProblem(), 15, 100, 0.5, 1) 
    method.run()
 
