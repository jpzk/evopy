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

from sklearn import svm
import numpy as np

class SVCLinearMetaModel:
    """ SVC meta model which classfies feasible and infeasible points """

    def train(self, feasible, infeasible, parameter_C = 1.0):
        """ Train a meta model classification with new points """

        points_svm = [i.value for i in infeasible] + [f.value for f in feasible]

        labels = [-1] * len(infeasible) + [1] * len(feasible) 
        self._clf = svm.SVC(kernel = 'linear', C = parameter_C)
        self._clf.fit(points_svm, labels)

    def repair(self, individual):
        x = individual.value

        w = self._clf.coef_[0]
        nw = w / np.sqrt(np.sum(w ** 2))
        b = self._clf.intercept_[0] / w[1]
      
        s = 2 * (self._clf.decision_function(x) * (1/np.sqrt(np.sum(w ** 2))))
        nx = x + (nw * s)

        individual.value = nx[0]
        return individual

    def check_feasibility(self, individual):
        """ Check the feasibility with meta model """

        prediction = self._clf.predict(individual.value) 
        if(prediction < 0):
            return False
        else:
            return True
