# Using the magic encoding
# -*- coding: utf-8 -*-

'''
This file is part of evopy.

Copyright 2012 - 2013, Jendrik Poloczek

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

from numpy import vsplit

from sys import path
from evopy.helper.logger import Logger
path.append("../..")

class SingleSimulator(object):
    name = "evopy: framework for experimention in evolutionary computing"
    description = "Single-Threaded Simulator for Continuous Constraint Handling"
    description_short = "SingleSimulator"

    def __init__(self, optimizer, problem, termination):
        self.optimizer = optimizer
        self.problem = problem
        self.termination = termination
        self.logger = Logger(self)

        self._count_cfc = 0
        self._count_ffc = 0
        self._generations = 0
        self.logger.add_binding('_count_cfc', 'count_cfc')
        self.logger.add_binding('_count_ffc', 'count_ffc')
        self.logger.add_binding('_generations', 'generations')

    def _information(self):
        print ("-" * 80) + "\n" + self.name +"\n" + ("-" * 80)
        print "simulator: " + self.description
        print "optimizer: " + self.optimizer.description
        print "problem: " + self.problem.description
        print "-" * 80

    def simulate(self):
        self._information()
        while(True):
            # SingleSimulator and optimizer handling constraints
            all_evaluated = False

            while(not all_evaluated):
                solutions = self.optimizer.ask_pending_solutions()

                info = lambda solution, position :\
                    (solution, self.problem.fitness(solution[0]),\
                    self.problem.penalty(position))

                informations = []
                for solution in solutions:
                    information = vsplit(solution, solution.shape[0])
                    position = information[0]
                    informations.append(info(solution, position))
                    self._count_cfc += 1
                    self._count_ffc += 1

                all_evaluated = self.optimizer.tell_fitness_penalty(informations)

            optimum, optimum_fitness = self.optimizer.ask_best_solution()

            # UPDATE OWN STATS
            self._generations += 1
            self.logger.log()
            self._count_cfc = 0
            self._count_ffc = 0
            print "%.20f" % (optimum_fitness)

            # TERMINATION
            if(self.termination.terminate(optimum_fitness, self._generations)):
                break
        return self
