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

from evopy.individuals.individual import Individual
from evopy.metamodel.svc_meta_model import SVCMetaModel
from evolution_strategy import EvolutionStrategy

class SVCEvolutionStrategy(EvolutionStrategy):

    _count_is_meta_feasible = 0
    _count_train_metamodel = 0

    _statistics_parameter_C_trajectory = []
    _statistics_parameter_gamma_trajectory = []
    _statistics_best_acc_trajectory = []

    _meta_model = SVCMetaModel()

    def __init__(\
        self, problem, mu, lambd, alpha, sigma, combination,\
        mutation, selection, view, parameter_C, parameter_gamma):

        super(SVCEvolutionStrategy, self).__init__(\
            problem, mu, lambd, alpha, sigma, combination,\
            mutation, selection, view)
       
        self._parameter_C = parameter_C
        self._parameter_gamma = parameter_gamma

    def log(\
        self, generation, next_population, best_acc, parameter_C,\
        parameter_gamma):
        
        super(SVCEvolutionStrategy, self).log(generation, next_population)

        self._statistics_best_acc_trajectory.append(best_acc)
        self._statistics_parameter_C_trajectory.append(parameter_C)
        self._statistics_parameter_gamma_trajectory.append(parameter_gamma)

    def view(\
        self, generation, next_population, best_acc, parameter_C,\
        parameter_gamma):
        
        self._view.view(generation, next_population, self._problem.fitness,\
            best_acc, parameter_C, parameter_gamma)

    def get_statistics(self):
        statistics = {
            "metamodel-calls" : self._count_is_meta_feasible,
            "train-function-calls" : self._count_train_metamodel,
            "best-acc" : self._statistics_best_acc_trajectory,
            "parameter-C" : self._statistics_parameter_C_trajectory,
            "parameter-gamma": self._statistics_parameter_gamma_trajectory}
        
        super_statistics = super(SVCEvolutionStrategy, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]
        
        return statistics

    # return true if solution is feasible in meta model, otherwise false.
    def is_meta_feasible(self, x):
        self._count_is_meta_feasible += 1
        return self._meta_model.check_feasibility(x)

    # train the metamodel with given points
    def train_metamodel(\
        self, 
        feasible, 
        infeasible,
        parameter_C,
        parameter_gamma):

        self._count_train_metamodel += 1
        self._meta_model.train(\
            feasible, 
            infeasible,
            parameter_C,
            parameter_gamma)

