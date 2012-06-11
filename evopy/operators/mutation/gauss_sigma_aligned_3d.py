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

class GaussSigmaAligned3D():
    def rotateX3(self, alpha):
        return matrix(\
            [[cos(alpha), -sin(alpha), 0],
            [sin(alpha), cos(alpha), 0],
            [0, 0, 1]])

    def rotateX2(self, alpha):
        return matrix(\
            [[cos(alpha), 0, sin(alpha)],
            [0, 1, 0],
            [-sin(alpha), 0, cos(alpha)]])

    def mutate(self, child, sigmas, hyperplane_normal):
        x = child.value
        d = len(x)

        # hyperplane alignment        
        inormal = -hyperplane_normal
        theta = arctan2(inormal[1], inormal[0])
        embedded_normal = (inormal * self.rotateX3(-theta)).getA1()
        phi = arctan2(embedded_normal[2], embedded_normal[1])

        # mutation
        m = rand(1,d) * transpose(array([normal(0, sigma) for sigma in sigmas]))
        m_r = transpose(m * self.rotateX2(phi) * self.rotateX3(-theta))

        new_x = array(x) + m_r
        child.value = new_x.tolist()[0]
        return child

