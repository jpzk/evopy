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
from numpy import array

from evolution_strategy import EvolutionStrategy

class DSES(EvolutionStrategy): 

    _statistics_parameter_epsilon_trajectory = []
    _statistics_DSES_infeasibles_trajectory = []
    _statistics_average_sigma_trajectory = []

    listadd = lambda self, l1, l2 : map(lambda i1, i2 : i1 + i2, l1, l2)
    meansigmas = lambda self, sigmas : map(lambda sigma : sigma / len(sigmas),\
        reduce(self.listadd, sigmas))

    def __init__(self, problem, mu, lambd, theta, pi, epsilon, tau0, tau1, \
        combination, mutation, selection, view, selfadaption):

        super(DSES, self).\
            __init__(problem, mu, lambd, \
            combination, mutation, selection, view) 

        # Death Penalty step control parameters
        self._theta = theta
        self._pi = pi
        self._epsilon = epsilon

        # Selfadaption           
        self._selfadaption = selfadaption
        self._tau0 = tau0
        self._tau1 = tau1

    def log(\
        self, generation, next_population, parameter_epsilon, DSES_infeasibles):
        
        super(DSES, self).log(generation, next_population)

        sigmas = map(lambda child : child.sigmas, next_population)

        self._statistics_average_sigma_trajectory.append(self.meansigmas(sigmas))
        self._statistics_parameter_epsilon_trajectory.append(parameter_epsilon)
        self._statistics_DSES_infeasibles_trajectory.append(DSES_infeasibles)

    def view(\
        self, generation, next_population, epsilon, DSES_infeasibles):

        sigmas = map(lambda child : child.sigmas, next_population)

        self._view.view(generation, next_population, self._problem.fitness,\
            epsilon, DSES_infeasibles, array(self.meansigmas(sigmas)).mean())

    def get_statistics(self):
        statistics = {
            "parameter-epsilon" : self._statistics_parameter_epsilon_trajectory,
            "DSES-infeasibles" : self._statistics_DSES_infeasibles_trajectory,
            "avg-sigma" : self._statistics_average_sigma_trajectory}
        
        super_statistics = super(DSES, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]
        
        return statistics

    # generate child 
    def generate_child(self, population, minimum_sigma):
        combined_child = self.combine(population)
        mutated_child = self.mutate(combined_child, combined_child.sigmas)
        selfadapted_child = self._selfadaption.mutate(\
            mutated_child, self._tau0, self._tau1)

        # minimum DSES step size control
        for sigma in selfadapted_child.sigmas: 
            if(sigma < minimum_sigma):
                sigma = minimum_sigma

        return selfadapted_child            

    def _run(self, (population, generation, m, l, lastfitness, epsilon)):

        DSES_infeasibles = 0

        feasible_children = []

        while(len(feasible_children) < l): 
            child = self.generate_child(population, epsilon) 
            if(self.is_feasible(child)):
                feasible_children.append(child)
            else:
                DSES_infeasibles += 1

        next_population = self.select(population, feasible_children, m)
        fitness_of_best = self.fitness(next_population[0])

        # step size reduction if infeasibles >= pi
        if(DSES_infeasibles >= self._pi):
            epsilon = epsilon * self._theta

        self.log(generation, next_population, epsilon, DSES_infeasibles)
        self.view(generation, next_population, epsilon, DSES_infeasibles)         

        if(self.termination(generation, fitness_of_best)):
            print next_population[0]
            return True
        else:
            return (next_population, generation + 1, m,\
            l, fitness_of_best, epsilon)

    def run(self):
        feasible_parents = []
        while(len(feasible_parents) < self._mu):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasible_parents.append(parent)

        result = self._run((feasible_parents, 0, self._mu,\
            self._lambd, 0, self._epsilon))

        while result != True:
            result = self._run(result)

        return result
