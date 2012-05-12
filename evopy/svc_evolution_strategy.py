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

from individual import Individual
from svc_meta_model import SVCMetaModel
from evolution_strategy import EvolutionStrategy

class SVCEvolutionStrategy(EvolutionStrategy):

    _count_is_meta_feasible = 0
    _count_train_metamodel = 0
    _count_sum_wrong_class = 0
    _meta_model = SVCMetaModel()

    def __init__(\
        self, problem, mu, lambd, alpha, sigma, parameter_C,\
        parameter_gamma):

        super(SVCEvolutionStrategy, self).__init__(\
            problem, mu, lambd, alpha, sigma)
       
        self._parameter_C = parameter_C
        self._parameter_gamma = parameter_gamma

    def get_statistics(self):
        return {
            "metamodel-calls" : self._count_is_meta_feasible,
            "train-function-calls" : self._count_train_metamodel,
            "sum-wrong-classification" : self._count_sum_wrong_class } 

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

