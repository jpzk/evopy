Unimodal Constrained Optimization
=================================

Because the ASK/TELL optimizer pattern is used, the optimizers are used as a component in a simulator. Different optimizers can be used to solve a certain problem. Please make sure you use the right simulator with a certain optimizer.  

Simple Death Penalty ES
-----------------------

.. autoclass:: evopy.strategies.bconstraint.simple_es.SimpleES
    :members:
This is a brief example of how to use the algorithm. ::

    from evopy.simulators.bconstraint.single_simulator import SingleSimulator
    from evopy.problems.oh_problem import OHProblem
    from evopy.strategies.bconstraint.simple_es import SimpleES
    from evopy.operators.termination.accuracy import Accuracy
    from numpy import matrix

    problem = OHProblem(dimensions = 5)
    start = matrix([[10.0] * 5])
    optimizer = SimpleES(\
        mu=15, lambd=100, rho=2, alpha=1.1, xstart=start, sigma = 4.5)
    termination = Accuracy(problem.optimum_fitness(), 10**(-3))
    sim = SingleSimulator(optimizer, problem, termination)
    sim.simulate()


Death Penalty Step Control ES
-----------------------------

Death Penalty Step Size Control Evolution Strategy proposed in Book_. It is based on a self-adaptive step size approach with minimum step size.

.. autoclass:: evopy.strategies.bconstraint.ori_dses.ORIDSES
    :members:

.. _Book: http://dl.acm.org/citation.cfm?id=1457333&coll=DL&dl=GUIDE&CFID=338588394&CFTOKEN=91607588

This is a brief example of how to use the algorithm. ::

    from evopy.simulators.bconstraint.single_simulator import SingleSimulator
    from evopy.problems.oh_problem import OHProblem
    from evopy.strategies.bconstraint.ori_dses import ORIDSES
    from evopy.operators.termination.accuracy import Accuracy
    from numpy import matrix

    problem = OHProblem(dimensions = 5)
    start = matrix([[10.0] * 5])
    optimizer = SimpleES(\
        mu=15, lambd=100, rho=2, alpha=1.1, xstart=start, sigma = 4.5)
    termination = Accuracy(problem.optimum_fitness(), 10**(-3))
    sim = SingleSimulator(optimizer, problem, termination)
    sim.simulate()

(:math:`\mu`, :math:`\lambda`)-CMA-ES with Death Penalty
--------------------------------------------------------

This strategy is based on the unconstrained (:math:`\mu`, :math:`\lambda`)-CMA-ES from Nikolaus Hansen. For more information on the base algorithm, check out the according Tutorial_. The constraint handling is based on the Death Penalty Approach. The evolution strategy can be used with binary constraint simulators. 

.. autoclass:: evopy.strategies.bconstraint.cmaes.CMAES
    :members:

.. _Wikipedia: http://www.wikipedia.org/
.. _Tutorial: https://www.lri.fr/~hansen/cmatutorial.pdf


(1+1)-CMA-ES for Constrained Optimization 
-----------------------------------------

This strategy is proposed in "A (1+1)-CMA-ES for Constrained Optimisation" by Dirk V. Arnold and Nikolaus Hansen, see Paper_. It is based on the underlying Cholesky-(1+1)-CMA-ES and extended by constraint vectors which approximate linear constraint boundaries. 

.. autoclass:: evopy.strategies.bconstraint.cmaes11.CMAES11
    :members:

.. _Paper: http://dl.acm.org/citation.cfm?id=2330163.2330207&coll=DL&dl=ACM&CFID=338588394&CFTOKEN=91607588


