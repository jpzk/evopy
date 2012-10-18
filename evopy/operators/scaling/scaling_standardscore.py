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

from numpy import array, matrix, vectorize
from copy import deepcopy

class ScalingStandardscore():
    """ Scaling to standardscore """

    def setup(self, individuals):
        dimensions = individuals[0].size
       
        iinrow = []
        for individual in individuals: 
            iinrow.append(individual.getA1())
        temp = matrix(iinrow)
        self._mean = temp.mean(axis=0)
        self._std = temp.std(axis=0)

    def scale(self, individualx):
        individual = deepcopy(individualx)
        val = individual
        dimensions = val.size

        scale = lambda val, mean, std : (val - mean) / std
        mat_scale = vectorize(scale)
          
        individual = mat_scale(individualx, self._mean, self._std)
        return individual

