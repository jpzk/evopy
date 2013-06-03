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

from random import random, sample, gauss
from numpy import vectorize

class OHProblem():

    description = "Sphere function with origin hyperplane restriction"
    description_short = "OH"

    def __init__(self, dim):
        self._d = dim

    def _power(self, x):
        _lpower = vectorize(lambda x : pow(x,2))
        return _lpower(x)

    def is_feasible(self, x):
        return float(x.sum()) >= 0

    def fitness(self, x):
        return self._power(x).sum()

    def optimum_fitness(self):
        return float(0.0)
