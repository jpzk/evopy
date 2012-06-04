from numpy import matrix, cos, sin, inner, array, sqrt, arccos, pi, arctan2
from numpy.random import rand
from numpy.random import normal

import pylab

hnormal = array([-1, -1])
hnormal2 = array([1, 1])

point = array([1, 0])
point2 = array([0,-1])

def calc_angle(normal):
    inormal = -normal
    return arctan2(inormal[0], inormal[1]) 

def norm(x):
    return sqrt(sum(x ** 2))

def rotate(alpha):
    return matrix([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])

points = [rand(2,1) for i in range(0,100)]
points = [array([p[0] * normal(0, 0.1), p[1] * normal(0,1)]) for p in points]

rotated = [array(rotate(calc_angle(hnormal)) * p) for p in points]

X = [point[0] for point in points]
Y = [point[1] for point in points]

Xr = [point[0] for point in rotated]
Yr = [point[1] for point in rotated]

print points
print rotated

pylab.axis([-2, 2, -2, 2])
pylab.plot(X,Y, "or")
pylab.plot(Xr, Yr, "og")

pylab.show()


