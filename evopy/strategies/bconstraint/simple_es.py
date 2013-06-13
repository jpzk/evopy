# Using the magic encoding
# -*- coding: utf-8 -*-

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

from numpy import array, random, matrix, exp, vectorize, mean
from numpy.random import normal, randn
from random import sample

from evopy.helper.logger import Logger

class SimpleES(object):
    """ This is a very simple ES with Rechenberg success rule """

    description = "A very simple ES with Rechenberg success rule"
    description_short = "Simple ES"

    def __init__(self, mu, lambd, rho, alpha, xstart, sigma):
        self._xstart = xstart
        self._sigma = sigma
        self._mu = mu
        self._rho = rho
        self._lambd = lambd
        self._alpha = alpha
        self._population = []
        self._valid_solutions = []
        rows, self._dimensions = self._xstart.shape

        self._best_fitness = None
        self._best_child = None

        self.logger = Logger(self)
        self._initialize_population()

    def _initialize_population(self):
        while(len(self._population) < self._mu):
            individual = matrix([self._sigma * randn(self._dimensions)])
            self._population.append(individual)

    def _generate_individual(self):
        parents = sample(self._population, self._rho)
        recombinated_child = mean(parents)
        mutation = matrix(self._sigma * random.randn(self._dimensions))
        mutated_child = recombinated_child + mutation
        return mutated_child

    def ask_pending_solutions(self):
        """ The simulator ask for produced solutions """
        return [self._generate_individual()]

    def tell_feasibility(self, feasibility_information):
        """ The simulator tells feasibility and the strategy
        handles the feasiblity of solutions. """
        for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)

        if(len(self._valid_solutions) < self._lambd):
            return False
        else:
            return True

    def ask_valid_solutions(self):
        """ The simulator asks for feasible solutions, the
        strategy returns valid solutios """

        return self._valid_solutions

    def tell_fitness(self, fitnesses):
        """ The simulator tells the optimizer the needed
        fitness information, the strategy handles the
        information and updates internal strategy parameters. """

        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        # success probability
        if(not self._best_fitness == None):
            better = lambda child : fitness(child) < self._best_fitness
            succprob = float(len(filter(better, fitnesses))) / self._lambd
            if(succprob < 0.2):
                self._sigma /= self._alpha
            else:
                self._sigma *= self._alpha

        # selection of survival
        sorted_fitnesses = sorted(fitnesses, key = fitness)[:self._mu]
        sorted_children = map(child, sorted_fitnesses)

        self.population = []
        for individual, fitness in sorted_fitnesses:
            self.population.append(individual)

        self._best_child, self._best_fitness = sorted_fitnesses[0]
        self._word_child, self._worst_fitness = sorted_fitnesses[-1]
        self.logger.log()

        self._valid_solutions = []

        return self._best_child, self._best_fitness

