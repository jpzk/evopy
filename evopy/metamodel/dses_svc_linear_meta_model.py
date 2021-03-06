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

from collections import deque

from sklearn import svm
from sklearn import __version__ as sklearn_version

from numpy import sum, sqrt, mean, arctan2, pi, matrix, sin, cos
from numpy import matrix, cos, sin, inner, array, sqrt, arccos, pi, arctan2
from numpy import transpose
from numpy.random import rand
from numpy.random import normal
from numpy.linalg import inv

from meta_model import MetaModel

class DSESSVCLinearMetaModel(MetaModel):
    """ DSES SVC meta model which classfies feasible and infeasible points """

    def __init__(self, window_size, scaling, crossvalidation, repair_mode):

        super(DSESSVCLinearMetaModel, self).__init__()

        self._window_size = window_size
        self._scaling = scaling            
        self._training_infeasibles = deque(maxlen = self._window_size)
        self._crossvalidation = crossvalidation
        self._repair_mode = repair_mode

        self.logger.add_binding('_selected_feasibles', 'selected_feasibles')
        self.logger.add_binding('_selected_infeasibles', 'selected_infeasibles')
        self.logger.add_binding('_best_acc', 'best_acc')
        self.logger.add_binding('_best_parameter_C', 'best_parameter_C')

    def is_trained():
        return self._trained

    def add_sorted_feasibles(self, feasibles):
        self._training_feasibles = feasibles

    def add_infeasible(self, infeasible):
        self._training_infeasibles.append(infeasible)

    def check_feasibility(self, individual):
        """ Check the feasibility with meta model """

        scaled_individual = self._scaling.scale(individual)
        prediction = self._clf.predict(scaled_individual.getA1())

        encode = lambda distance : False if distance < 0 else True
        return encode(prediction)
        
    def train(self):
        """ Train a meta model classification with new points, return True
            if training was successful, False if not enough infeasible points 
            are gathered """

        if(len(self._training_infeasibles) < self._window_size):
            self._selected_feasibles = None
            self._selected_infeasibles = None
            self._best_parameter_C = None
            self._best_acc = None
            self._normal = None
            self.logger.log()
            return False

        cv_feasibles = self._training_feasibles[:self._window_size]
        cv_infeasibles = [inf for inf in self._training_infeasibles]

        self._scaling.setup(cv_feasibles + cv_infeasibles)

        scale = lambda child : self._scaling.scale(child)
        scaled_cv_feasibles = map(scale, cv_feasibles)
        scaled_cv_infeasibles = map(scale, cv_infeasibles)

        self._selected_feasibles, self._selected_infeasibles,\
        self._best_parameter_C, self._best_acc =\
            self._crossvalidation.crossvalidate(\
                scaled_cv_feasibles, scaled_cv_infeasibles)

        # @todo WARNING maybe rescale training feasibles/infeasibles (!) 
        fvalues = [f.getA1() for f in self._selected_feasibles]
        ivalues = [i.getA1() for i in self._selected_infeasibles]

        points = ivalues + fvalues
        labels = [-1] * len(ivalues) + [1] * len(fvalues) 

        self._clf = svm.SVC(\
            kernel = 'linear',\
            C = self._best_parameter_C, \
            tol = 1.0)
        self._clf.fit(points, labels)

        self.logger.log()
        return True

    def distance_to_hp(self, x):
        """ Returns the distance from a point to hyperplane """
        return (self._clf.decision_function(x) * (1/sqrt(sum(w ** 2))))

    def get_normal(self):
        # VERY IMPORTANT
        w = self._clf.coef_[0]
        nw = w / sqrt(sum(w ** 2))
 
        if sklearn_version == '0.10':
            return -nw 
        if sklearn_version == '0.11':
            return nw 
        if sklearn_version != '0.10' and sklearn_version != '0.11':
            raise Exception("sklearn version is not supported")

    def repair(self, individual, sigma):

        x = self._scaling.scale(individual)
        w = self._clf.coef_[0]
        nw = self.get_normal()
        b = self._clf.intercept_[0] / w[1]
     
        to_hp = (self._clf.decision_function(x) * (1/sqrt(sum(w ** 2))))
        if self._repair_mode == 'mirror':
            s = 2 * to_hp
        if self._repair_mode == 'none':
            return individual
        if self._repair_mode == 'project':
            s = to_hp 
        if self._repair_mode == None: 
            raise Exception("no repair_mode selected: " + repair_mode)

        nx = x + (-nw * s)  #(-nw * (1/sqrt(sum(w ** 2))))
        return (self._scaling.descale(nx)) #+ (-nw * sigma.min())

