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

from numpy import array, matrix, vectorize
from copy import deepcopy

class ScalingStandardscore():
    """ Scaling to standardscore """

    def setup(self, values):
        dimensions = values[0].size
       
        iinrow = []
        for val in values: 
            iinrow.append(val.getA1())
        temp = matrix(iinrow)
        self._mean = temp.mean(axis = 0)
        self._std = temp.std(axis = 0)

    def scale(self, valx):
        scale = lambda val, mean, std : (val - mean) / std if std != 0 else val
        _mat_scale = vectorize(scale) 
        return _mat_scale(valx, self._mean, self._std)   

    def descale(self, valx):
        descale = lambda val, mean, std : (val * std) + mean if std != 0 else val
        _mat_descale = vectorize(descale) 
        return _mat_descale(valx, self._mean, self._std)
