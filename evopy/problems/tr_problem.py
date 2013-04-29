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

from numpy import vectorize, float64

class TRProblem():

    description = "Sphere function with tangent restriction"
    description_short = "TR"

    def __init__(self, dimensions = 2, size = 10):
        self._d = dimensions
        self._size = 10

    def _power(self, x):
        _lpower = vectorize(lambda x : pow(x,2))
        return _lpower(x)

    def is_feasible(self, x):
        return float64(x.sum()) - float64(self._d) >= 0

    def penalty(self, x):
        return max(0, -(x.sum() - self._d))

    def fitness(self, x):
        return self._power(x).sum()

    def optimum_fitness(self):
        return float64(self._d)
