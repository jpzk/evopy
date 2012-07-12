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
   
    def __init__(self, mu, lambd):
        self._mu = mu
        self._lambd = lambd

        self._statistics_best_fitness_trajectory = []
        self._statistics_worst_fitness_trajectory = []
        self._statistics_mean_fitness_trajectory = []
        self._statistics_selected_children_trajectory = []

    def log(self, generation, next_population):
        self._statistics_generations += 1
        #fitnesses = array(map(self._problem.fitness, next_population))
        #self._statistics_worst_fitness_trajectory.append(fitnesses.max())
        #self._statistics_average_fitness_trajectory.append(fitnesses.mean())
        #self._statistics_best_fitness_trajectory.append(fitnesses.min())

    def get_last_statistics(self):
        return {
            "best_fitness": self._statistics_best_fitness_trajectory[-1],
            "worst_fitness": self._statistics_worst_fitness_trajectory[-1],
            "avg_fitness": self._statistics_mean_fitness_trajectory[-1],
            "selected_children": self._statistics_selected_children_trajectory[-1]}
   
    def get_statistics(self):
        return {
            "best_fitness": self._statistics_best_fitness_trajectory,
            "worst_fitness": self._statistics_worst_fitness_trajectory,
            "avg_fitness": self._statistics_mean_fitness_trajectory,
            "selected_children": self._statistics_selected_children_trajectory}

