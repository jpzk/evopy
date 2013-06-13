Documentation of evopy
======================

The evopy project is a framework for experimenting with evolutionary algorithms (EA). The framework is based on widely used libraries like SciPy [1]_, NumPy [1]_ and Matplotlib [2]_. In contrast to other EA frameworks for Python, like `PyGMO`_, `inspyred`_ and `DEAP`_ [3]_, evopy focusses on new algorithms and provides tools for analysis and comparison of different algorithms. Furthermore, evopy concentrates on (constrained) *numerical* black box optimization with EA. 

.. _PyGMO : http://pagmo.sourceforge.net/pygmo/index.html
.. _inspyred : http://inspyred.github.io/
.. _DEAP : https://code.google.com/p/deap/

Brief Example
-------------

In this example, we use the single-threaded simulator for binary constraints, the tangent-restriction problem and the (1+1)-CMA-ES for constrained optimization. As termination condition we use the accuracy termination operator. ::


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
    unconstrained
    constrained

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] Travis E. Oliphant (2007).  Python for Scientific Computing. Computing in Science & Engineering 9, IEEE Soc.
.. [2] Hunter, J.  D. (2007). Matplotlib: A 2D graphics environment. Computing In Science & Engineering 9, IEEE Soc., pp. 90-95
.. [3] Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 2171-2175, no 13, jul 2012.
.. [4] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and R. Le Riche (2010). On Object-Oriented Programming of Optimizers - Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.: Multidisciplinary Design Optimization in Computational Mechanics, Wiley, pp. 527-565;


