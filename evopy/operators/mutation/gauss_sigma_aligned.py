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

from random import gauss

from numpy import matrix, cos, sin, inner, array, sqrt, arccos, pi, arctan2
from numpy import transpose
from numpy.random import rand
from numpy.random import normal

class GaussSigmaAligned():
    def mutate(self, child, sigmas, hyperplane_normal):
        x = child.value
        d = len(x)

        # hyperplane alignment        
        inormal = -hyperplane_normal
        rad = arctan2(inormal[0], inormal[1])
        rotate = matrix([[cos(rad), -sin(rad)], [sin(rad), cos(rad)]])

        # mutation
        m = rand(1,d) * transpose(array([normal(0, sigma) for sigma in sigmas]))
        m_r = transpose(rotate * transpose(m))

        new_x = array(x) + m_r
        child.value = new_x.tolist()[0]
        return child

