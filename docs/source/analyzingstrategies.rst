Analyzing an Optimizer
======================

Using the Logger on built-in Optimizers
---------------------------------------

Here comes a brief example of how to use evopy::

    simulator = SingleSimulator(optimizer, problem, termination)
    simulator.simulate()
    simulator.optimizer.logger.all()
