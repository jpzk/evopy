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

class SchwefelsProblem26:

    description = "Schwefel's problem 2.6"
    description_short = "Schwefel26"

    def __init__(self, dimensions = 2, size = 100):
        self._d = dimensions
        self._size = size

    def is_feasible(self, x):
        return sum(x.value) - 70 >= 0

    def fitness(self, x):
        v = x.value
        return max(abs(v[0] + 2 * v[1] - 7), abs(2 * v[0] + v[1] - 5))
                               
    def optimum_fitness(self):
        return 0.0 
