Getting Started 
===============

Understanding the ASK/TELL Pattern
----------------------------------

Using a Built-In Optimizer
--------------------------

Here comes a brief example of how to use evopy::

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



Analyzing a Built-In Optimizer
------------------------------

.. plot:: pyplots/fitness.py

