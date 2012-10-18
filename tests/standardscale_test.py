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
from numpy import matrix
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore

def standardscore_scaling_test():
    i1, i2 = matrix([[1.0, 1.0]]), matrix([[2.0, 2.0]])
    individuals = [i1, i2]

    scaling = ScalingStandardscore()
    scaling.setup(individuals)
    m = scaling.scale(i1)

    assert True
