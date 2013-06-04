from numpy import array
from itertools import product, groupby
from copy import deepcopy

class Sampling(object):

    def _run_value(self, value, parameter, args, algorithm):
        args[parameter] = value
        error = algorithm(args)
        algorithm = None
        return (value, error)

    def sample(self, parameters, intervals, stepsizes, args, algorithm, parallel=True):

        steps = {}
        values = {}
        results = []

        for parameter, interval, stepsize in zip(parameters, intervals, stepsizes):
            diff = interval[1] - interval[0]
            if(diff % stepsize > 0):
                raise Exception("Steps dont fit into interval")
            steps[parameter] = int(diff / stepsize)
            values[parameter] = []
            for step in xrange(steps[parameter]):
                value = interval[0] + step * stepsize
                values[parameter].append(value)

        iterables = []
        for k, vals in values.iteritems():
            iterables.append([(k, v) for v in vals])

        for combination in product(*iterables):
            xargs = deepcopy(args)
            for parameter in combination:
                name, value = parameter
                xargs[name] = value

            results.append((xargs, algorithm(xargs)))

        return results


"""
args = {}
parameters = ['a','b']
intervals = [[-10, 10], [-10, 10]]
stepsizes = [1,1]

def f(args):
    return args['a'] * args['b']

sam = Sampling()
res = sam.sample(parameters, intervals, stepsizes, args, f)

values, steps = {}, {}
for parameter, interval, stepsize in zip(parameters, intervals, stepsizes):
    diff = interval[1] - interval[0]
    if(diff % stepsize > 0):
        raise Exception("Steps dont fit into interval")
    steps[parameter] = int(diff / stepsize)
    values[parameter] = []
    for step in xrange(steps[parameter]):
        value = interval[0] + step * stepsize
        values[parameter].append(value)

A = array([values['a']] * len(values['a']))
B = array([values['b']] * len(values['b'])).T
C = []

for rowA, rowB in zip(A,B):
    C_row = []
    for valA, valB in zip(rowA, rowB):
        valC = filter(lambda h : h[0] == {'a':valA, 'b':valB}, res)[0][1]
        C_row.append(valC)
    C.append(C_row)

C = array(C)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel("A")
plt.ylabel("B")

wframe = ax.plot_wireframe(A, B, C, rstride=1, cstride=1)
plt.show()
"""
