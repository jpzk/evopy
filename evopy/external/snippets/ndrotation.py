from random import gauss

from numpy import matrix, cos, sin, inner, array, sqrt, arccos, pi, arctan2
from numpy import transpose
from numpy.linalg import inv
from numpy.random import rand
from numpy.random import normal

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

    def rotations(self, normal):       
        d = len(normal)
        rotations = []
        enormals = [transpose(normal)]
        for x, y in [(0, i) for i in range(1,3)]:
            lnormal = enormals[-1]
            print "lnormal", lnormal
            lnormal_as_list = lnormal.getA1()
            angle = arctan2(lnormal_as_list[y], lnormal_as_list[x])
            print x,y,angle * 180 / pi
            rotation = self.givens(x,y, -angle, 3)
            enormals.append(rotation * lnormal)
            rotations.append(rotation)
        rotations.reverse()            
        return rotations

inormal = matrix([[1, 1, 1]])

ga = GaussSigmaAlignedND();
rotations = ga.rotations(inormal)

# mutation
m = transpose(matrix([[1,0,0]]))

m_i = m
for rotation in rotations:
    m_i = inv(rotation) * m_i

m_r = transpose(m_i)

print m_r

