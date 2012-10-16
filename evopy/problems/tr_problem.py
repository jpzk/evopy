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

from numpy import vectorize

class TRProblem():

    description = "Sphere function with tangent restriction"
    description_short = "TR"

    def __init__(self, dimensions = 2, size = 10):
        self._d = dimensions
        self._size = 10

    def is_feasible(self, x):
        return x.sum() - float(self._d) >= 0

    def fitness(self, x):
        return (vectorize(lambda x : pow(x,2))(x)).sum()

    def optimum_fitness(self):
        return float(self._d)
