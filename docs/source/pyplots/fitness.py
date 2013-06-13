from sys import path
from pylab import *

path.append("../../..")

from evopy.simulators.bconstraint.single_simulator import SingleSimulator
from evopy.problems.tr_problem import TRProblem
from evopy.strategies.bconstraint.cmaes11 import CMAES11
from evopy.operators.termination.accuracy import Accuracy
from numpy import matrix

dimensions = 2
problem = TRProblem(dimensions)
optimizer = CMAES11(xstart = matrix([[10.0] * dimensions]), sigma = 4.5)
termination = Accuracy(problem.optimum_fitness(), 10**(-8))
simulator = SingleSimulator(optimizer, problem, termination)
simulator.simulate()

Y = simulator.logger.all()['best_fitness']
logY = map(lambda y : log10(abs(problem.optimum_fitness() - y)), Y)
X = range(len(Y))
xlabel("Generations")
ylabel("Best Fitness in Log10")
xlim([0, len(X)])
plot(X, logY, color='k')
show()

