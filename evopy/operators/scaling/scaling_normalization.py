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

class ScalingNormalization():
    """ Scaling to [0, 1] """

    def setup(self, values):
        dimensions = values[0].size

        iinrow = []
        for value in values: 
            iinrow.append(value.getA1())
        temp = matrix(iinrow)
        self._min = temp.min(axis = 0)
        self._max = temp.max(axis = 0)

    def scale(self, valx):
        value = deepcopy(valx)

        scale = lambda val, emin, emax : (2*(val - emin) / (emax - emin)) - 1
        mat_scale = vectorize(scale)

        value = mat_scale(valx, self._min, self._max)
        return value
