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

from numpy import array

class EvolutionStrategy(object):
   
    def __init__(self, problem, mu, lambd, \
        combination, mutation, selection, view, sigmas = None):

        self._problem = problem
        self._mu = mu
        self._lambd = lambd
        self._sigmas = sigmas
        self._mutation = mutation
        self._combination = combination
        self._selection = selection
        self._view = view

        self._statistics_is_feasible = 0
        self._statistics_fitness = 0
        self._statistics_generations = 0
        self._statistics_mutations = 0
        self._statistics_best_fitness_trajectory = []
        self._statistics_worst_fitness_trajectory = []
        self._statistics_average_fitness_trajectory = []

    def log(self, generation, next_population):
        self._statistics_generations += 1
        fitnesses = array(map(self._problem.fitness, next_population))
        self._statistics_worst_fitness_trajectory.append(fitnesses.max())
        self._statistics_average_fitness_trajectory.append(fitnesses.mean())
        self._statistics_best_fitness_trajectory.append(fitnesses.min())

    def get_statistics(self):
        return {
            "generations" : self._statistics_generations,
            "constraint-calls" : self._statistics_is_feasible,
            "fitness-function-calls" : self._statistics_fitness,
            "mutation-calls" : self._statistics_mutations,
            "best-fitness": self._statistics_best_fitness_trajectory,
            "worst-fitness": self._statistics_worst_fitness_trajectory,
            "avg-fitness": self._statistics_average_fitness_trajectory}

    def termination(self, generations, fitness_of_best):
        return self._problem.termination(generations, fitness_of_best)

    # return success_probabilty (rechenberg)
    def success_probability(self, children, success_fitness):
        return len(filter(
            lambda child: self.fitness(child) <= success_fitness, 
            children)) / len(children)

    # return true if solution is valid, otherwise false.
    def is_feasible(self, x):
        self._statistics_is_feasible += 1
        return self._problem.is_feasible(x) 

    def fitness(self, x):
        self._statistics_fitness += 1
        return self._problem.fitness(x)

    def view(self, generation, next_population):
        self._view.view(generation, next_population, self._problem.fitness)

    def combine(self, population):
        return self._combination.combine(population)

    # mutate child with gauss devriation 
    def mutate(self, child, sigmas):
        self._statistics_mutations += 1
        return self._mutation.mutate(child, sigmas)
      
    def select(self, population, children, mu):
        return self._selection.select(self.fitness, population, children, mu)

    def generate_population(self):
        generator = self._problem.population_generator() 
        return generator.next()

