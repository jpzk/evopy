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
from numpy.linalg import inv

class GaussSigmaAlignedND():

    def calculate_amount_planes(self, d):
        return (d * (d - 1))/2

    def givens(self, i, j, alpha, d):
        mat = []
        for a in range(0, d):
            row = []
            for b in range(0, d):
                if a == i and b == i:
                    row.append(cos(alpha))
                elif a == j and b == j:
                    row.append(cos(alpha))
                elif a == j and b == i:
                    row.append(sin(alpha))
                elif a == i and b == j:
                    row.append(-sin(alpha))                    
                elif a == b and a != j and b != j: 
                    row.append(1)   
                else:
                    row.append(0)
            mat.append(row)                                                
        return matrix(mat)

    def rotations(self, normal, d):       
        rotations = []
        enormals = [transpose(normal)]
        for x, y in [(0, i) for i in range(1,d)]:
            lnormal = enormals[-1]
            lnormal_as_list = lnormal.getA1()
            angle = arctan2(lnormal_as_list[y], lnormal_as_list[x])
            rotation = self.givens(x,y, -angle, d)
            enormals.append(rotation * lnormal)
            rotations.append(rotation)
        rotations.reverse()            
        return rotations

    def mutate(self, child, sigmas, hyperplane_normal):

        x = child.value
        d = len(x)

        # hyperplane alignment        
        inormal = -hyperplane_normal
        rad = arctan2(inormal[0], inormal[1])

        inormal = matrix(inormal)
        rotations = self.rotations(inormal, d)

        # mutation
        m = rand(1,d) * transpose(array([normal(0, sigma) for sigma in sigmas]))

        m_i = transpose(m)
        for rotation in rotations:
            m_i = inv(rotation) * m_i

        m_r = transpose(m_i)

        new_x = array(x) + m_r
        child.value = new_x.tolist()[0]
        return child

