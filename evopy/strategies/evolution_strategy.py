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

class EvolutionStrategy(object):
   
    _count_is_feasible = 0
    _count_fitness = 0
    _count_generations = 0
    _count_mutations = 0
    _count_children_generated = 0
    _count_population_generated = 0

    def __init__(self, problem, mu, lambd, alpha, sigma):

        self._problem = problem
        self._mu = mu
        self._lambd = lambd
        self._alpha = alpha
        self._sigma = sigma

    def _run(self, population, generation, mu, lambd, lastfit, alpha, sigma):
        self._count_generations += 1

    def get_statistics(self):
        return {
            "generations" : self._count_generations,
            "constraint-calls" : self._count_is_feasible,
            "fitness-function-calls" : self._count_fitness,
            "mutation-calls" : self._count_mutations,
            "children-generated" : self._count_children_generated,
            "pop-generated": self._count_population_generated}

    def termination(self, generations, fitness_of_best):
        return self._problem.termination(generations, fitness_of_best)

    # return success_probabilty (rechenberg)
    def success_probability(self, children, success_fitness):
        return len(filter(
            lambda child: self.fitness(child) <= success_fitness, 
            children)) / len(children)

    # return true if solution is valid, otherwise false.
    def is_feasible(self, x):
        self._count_is_feasible += 1
        return self._problem.is_feasible(x) 

    def fitness(self, x):
        self._count_fitness += 1
        return self._problem.fitness(x)

    # return combined child of parents x,y
    def combine(self, pair):
        self._count_combinations += 1
        return self._problem.combine(pair)

    # mutate child with gauss devriation 
    def mutate(self, child, sigma):
        self._count_mutations += 1
        self._problem.mutate(child, sigma)
      
    def sortedbest(self, children):
        return self._problem.sortedbest(children)

    def generate_population(self):
        self._count_population_generated += 1
        generator = self._problem.population_generator() 
        return generator.next()

    def generate_child(self, parents, sigma):
        self._count_children_generated += 1
        generator = self._problem.children_generator(parents, sigma)
        return generator.next()

