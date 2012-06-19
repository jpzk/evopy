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

    def get_angles_degree(self):
        return self.angles

    def rotations(self, normal, d):       
        rotations = []
        self.angles = []            
        enormals = [transpose(normal)]

        for x, y in [(0, i) for i in range(1,d)]:
            lnormal = enormals[-1]
            lnormal_as_list = lnormal.getA1()

            # calculate radian of last embedded normal
            angle = arctan2(lnormal_as_list[y], lnormal_as_list[x])

            # append angles for info
            self.angles.append(angle * 180.0/pi)

            # embed normal into next axis combination
            rotation = self.givens(x,y, -angle, d)

            # append embedded normal
            enormals.append(rotation * lnormal)

            # append rotation
            rotations.append(rotation)
        rotations.reverse()            
        return rotations

    def prepare_inverse_rotations(self, hyperplane_normal):
        inormal = -hyperplane_normal
        d = len(inormal)
        inormal = matrix(inormal)
        rotations = self.rotations(inormal, d)
        self.inverse_rotations = []

        for rotation in rotations:
            # transpose(rotation matrix) is inverse
            inv_rotation = transpose(rotation)
            self.inverse_rotations.append(inv_rotation)

        # left-associative reduce (important!)
        self.new_basis = reduce(lambda r1, r2 : r1 * r2, self.inverse_rotations)

    def mutate(self, child, sigmas):
        x = child.value
        d = len(x)

        # mutation
        m = rand(1,d) * transpose(array([normal(0, sigma) for sigma in sigmas]))
        m_i = self.new_basis * transpose(m)
        m_r = transpose(m_i)

        new_x = array(x) + m_r
        child.value = new_x.tolist()[0]
        return child

