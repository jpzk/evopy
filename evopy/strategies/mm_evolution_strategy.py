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
from evolution_strategy import EvolutionStrategy

class MMEvolutionStrategy(EvolutionStrategy):

    def __init__(self, mu, lambd, combination, mutation,\
        selection, view):

        super(MMEvolutionStrategy, self).__init__(\
            mu, lambd, combination, mutation, selection, view)

        self._count_is_meta_feasible = 0
        self._count_train_metamodel = 0
        self._statistics_best_acc_trajectory = []
        self._statistics_wrong_meta_infeasibles = []        

    def log(self, generation, next_population, best_acc, wrong_meta_infeasibles):        
        super(MMEvolutionStrategy, self).log(generation, next_population)

        self._statistics_best_acc_trajectory.append(best_acc)
        self._statistics_wrong_meta_infeasibles.append(wrong_meta_infeasibles)

    def get_statistics(self):
        statistics = {
            "metamodel-calls" : self._count_is_meta_feasible,
            "train-function-calls" : self._count_train_metamodel,
            "best-acc" : self._statistics_best_acc_trajectory,
            "wrong-meta-infeasibles" : self._statistics_wrong_meta_infeasibles}
        
        super_statistics = super(MMEvolutionStrategy, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]
        
        return statistics

