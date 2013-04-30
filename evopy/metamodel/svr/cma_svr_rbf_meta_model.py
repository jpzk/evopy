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

import sys

sys.path.append("../../../")

from collections import deque

from sklearn.svm import SVR
from sklearn import __version__ as sklearn_version

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

from sklearn.cross_validation import KFold
from numpy import sum, sqrt, mean, arctan2, pi, matrix, sin, cos
from numpy import matrix, cos, sin, inner, array, sqrt, arccos, pi, arctan2
from numpy import transpose
from numpy.random import rand
from numpy.random import normal
from numpy.linalg import inv

from evopy.metamodel.meta_model import MetaModel

class CMASVRRBFMetaModel(MetaModel):

    def __init__(self, window_size, scaling):

        super(CMASVRRBFMetaModel, self).__init__()

        self._window_size = window_size
        self._scaling = scaling

        self.logger.add_binding('_best_acc', 'best_acc')
        self.logger.add_binding('_best_parameter_C', 'best_parameter_C')
        self.logger.add_binding('_best_parameter_gamma',\
            'best_parameter_gamma')

        self._trained = False

    def is_trained():
        return self._trained

    def add(self, solutions):
        self._trainingset = solutions[:self._window_size]

    def predict(self, x):
        return self._svr.predict(x)[0]

    def train(self):

        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        trainingset = self._trainingset[:self._window_size]
        trainingset_child = map(child, trainingset)
        trainingset_fitness = map(fitness, trainingset)

        self._scaling.setup(map(child, trainingset))

        scale = lambda child : self._scaling.scale(child)
        scaled_trainingset_child = map(scale, trainingset_child)

        cv_method = KFold(self._window_size, 2)
        gamma_range = [0.0001, 0.000001, 0.0000001]
        C_range = [2 ** i for i in range(-3, 14, 2)]

        tuned_parameters = [{
            'kernel': ['rbf'],
            'C': C_range,
            'gamma': gamma_range}]

        # Training set
        X = map(lambda mat : mat.getA1(), scaled_trainingset_child)
        y = trainingset_fitness

        grid = GridSearchCV(SVR(kernel='rbf', epsilon = 0.1),
                param_grid = tuned_parameters, cv=cv_method, verbose=0)

        grid.fit(X, array(y))

        self._best_parameter_C = grid.best_estimator.C
        self._best_parameter_gamma = grid.best_estimator.gamma
        self._best_acc = grid.best_score

        self._svr = SVR(kernel='rbf', epsilon=0.1, C = self._best_parameter_C,\
            gamma = self._best_parameter_gamma)

        self._svr.fit(X, y)

        self.logger.log()
        self._trained = True
        return True

