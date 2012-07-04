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

import pdb
from numpy import array
from copy import deepcopy

class ScalingStandardscore():
    """ Scaling to standardscore """

    def setup(self, individuals):
        values = map(lambda i : i.value, individuals)
        dimensions = len(values[0])
       
        self._mean = []
        self._std = []
        for d in range(0, dimensions):
            vals = map(lambda u : u[d], values)
            self._mean.append(array(vals).mean())
            self._std.append(array(vals).std())

    def scale(self, individualx):
        individual = deepcopy(individualx)
        val = individual.value
        dimensions = len(val)
    
        scaled_value = []
        for d in range(0, dimensions):
            old_value = val[d]
            scaled_value.append((old_value - self._mean[d])/ self._std[d])

        individual.value = scaled_value
        return individual

