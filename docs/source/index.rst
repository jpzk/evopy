Documentation of evopy
======================

The evopy project is a framework for experimenting with evolutionary algorithms (EA). The framework is based on widely used libraries like SciPy, NumPy and Matplotlib. In contrast to other EA frameworks for Python, evopy focusses on new algorithms and provides tools for analysis and comparison of different algorithms. Besides, the architecture is based on the original ASK/TELL optimization pattern by X and an extended version for constrained optimization. At the moment only evolution strategies are implemented. 

Quick Example
-------------
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

Getting Started
---------------

.. toctree::
    about
    installation
    gettingstarted

Optimizers
----------

Because the ASK/TELL optimizer pattern is used, the optimizers are used as a component in a simulator. Different optimizers can be used to solve a certain problem. Please make sure you use the right simulator with a certain optimizer.  

Constrained Optimization
------------------------

.. toctree::
    dses
    cmaesdp
    cmaes11dp

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

