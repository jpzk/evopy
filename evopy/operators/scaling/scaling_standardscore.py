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
from copy import deepcopy

class ScalingStandardscore():
    """ Scaling to standardscore """

    def setup(self, individuals):
        values = map(lambda i : i.value, individuals)
        iarray = array(values)
        self._mean = iarray.mean()
        self._std = iarray.std()

    def scale(self, individualx):
        individual = deepcopy(individualx)
        val = individual.value
        scaled = map(lambda x : (x - self._mean) / self._std, val)
        individual.value = scaled        
        return individual 
