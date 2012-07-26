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

class SchwefelsProblem12():
    
    description = "Schwefel's problem 1.2"
    description_short = "Schwefel12"

    def __init__(self, dimensions = 2, size = 100):
        self._d = dimensions
        self._size = 10 

    def is_feasible(self, x):
        return x.value[0] - 50 >= 0

    def fitness(self, x):
        outer_sum = 0
        for d in range(0, self._d):
            inner_sum = 0
            for j in range(0, d):
                inner_sum += x.value[j]
            quadratic_inner_sum = inner_sum ** 2
            outer_sum += quadratic_inner_sum
        return outer_sum            
                                  
    def optimum_fitness(self):
        return 2500.0
