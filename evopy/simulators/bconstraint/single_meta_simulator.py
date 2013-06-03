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
from numpy.random import randn
from collections import deque
from evopy.metamodel.active.lahmce_simple import LAHMCESimple
from evopy.metamodel.active.active_plane import ActivePlane

from sys import path
from evopy.helper.logger import Logger

path.append("../..")

class SingleMetaSimulator(object):
    name = "evopy: framework for experimention in evolutionary computing"
    description = "Single-Threaded Meta Simulator for Binary Constraint Handling"
    description_short = "SingleMetaSimulator"

    def __init__(self, optimizer, problem, termination, budget):
        self.optimizer = optimizer
        self.problem = problem
        self.termination = termination
        self.logger = Logger(self)
        self.budget = budget

        self._count_cfc = 0
        self._cum_count_cfc = 0
        self._count_ffc = 0
        self._generations = 0

        self.logger.add_binding('_cum_count_cfc', 'cum_count_cfc')
        self.logger.add_binding('_count_cfc', 'count_cfc')
        self.logger.add_binding('_count_ffc', 'count_ffc')
        self.logger.add_binding('_generations', 'generations')

        self._spent = 0
        self.last_feasibles = deque(maxlen=self.problem._d)
        self.last_infeasibles = deque(maxlen=self.problem._d)
        self.plane = None
        self.trained = False

    def _information(self):
        print ("-" * 80) + "\n" + self.name +"\n" + ("-" * 80)
        print "simulator: " + self.description
        print "optimizer: " + self.optimizer.description
        print "problem: " + self.problem.description
        print "-" * 80

    def update_active_plane(self):

        last = zip(self.last_feasibles, self.last_infeasibles)
        points = []

        for feasible, infeasible in last:
            toadd = 2
            budget = self.budget

            linem = LAHMCESimple(feasible, infeasible, toadd, budget)
            mean, nf, ni, spent = linem.train(self.problem)
            points.append((nf, ni))
            self._spent += spent

        self.plane = ActivePlane(points)

    def simulate(self):
        self._information()

        while(True):
            # SingleSimulator and optimizer handling constraints
            all_feasible = False
            while(not all_feasible):
                # ASK for solutions (feasbile and infeasible)
                solutions = self.optimizer.ask_pending_solutions()

                # CHECK solutions for feasibility
                feasibility =\
                    lambda solution, position :\
                        (solution, self.problem.is_feasible(position))

                feasibilitym =\
                    lambda solution, position :\
                        (solution, self.plane.predict(position))

                feasibility_information = []
                for solution in solutions:
                    information = vsplit(solution, solution.shape[0])
                    position = information[0]

                    if(not self.trained):
                        f = feasibility(solution, position)
                        feasibility_information.append(f)
                        self._count_cfc = 1
                        if(f[1]):
                            self.last_feasibles.append(f[0])
                        else:
                            self.last_infeasibles.append(f[0])

                        # check if enough last fea. inf.
                        if(len(self.last_feasibles) == self.problem._d and
                            len(self.last_infeasibles) == self.problem._d):
                            self.update_active_plane()
                            self.trained = True
                    else:
                        if(self.plane.predictable(position)):
                            f = feasibilitym(solution, position)
                            feasibility_information.append(f)
                            tf = feasibility(solution, position)
                            if(f[1] != tf[1]):
                                raise Exception("active plane incorrect")
                        else:
                            print "Updating Active Plane"
                            # sample with original cfc function
                            self.last_feasibles.clear()
                            self.last_infeasibles.clear()
                            self.trained = False

                # TELL feasibility, returns True if all feasible,
                # returns False if extra checks
                all_feasible =\
                    self.optimizer.tell_feasibility(feasibility_information)

            # ASK for valid solutions (feasible)
            valid_solutions = self.optimizer.ask_valid_solutions()

            # CHECK fitness
            fitnesses = []
            fitness = lambda solution : (solution, self.problem.fitness(solution[0]))
            for solution in valid_solutions:
                fitnesses.append(fitness(solution))
                self._count_ffc += 1

            # TELL fitness, return optimum
            optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)

            # A-POSTERIORI information for confusion matrix
            if('ask_a_posteriori_solutions' in dir(self.optimizer)):
                apos_feasibility =\
                    lambda (position, meta_feasibility) :\
                    (position, meta_feasibility, self.problem.is_feasible(position))

                apos_solutions = self.optimizer.ask_a_posteriori_solutions()
                feasibility_info = []
                for solution in apos_solutions:
                    information = vsplit(solution[0], solution[0].shape[0])
                    position = information[0]
                    meta_feasibility = solution[1]
                    feasibility_info.append(apos_feasibility((position, meta_feasibility)))

                self.optimizer.tell_a_posteriori_feasibility(feasibility_info)

            # UPDATE OWN STATS
            self._generations += 1
            self._cum_count_cfc += self._count_cfc

            self.logger.log()
            self._count_cfc = 0
            self._count_ffc = 0
            print optimum, "%.20f" % (optimum_fitness)

            # TERMINATION
            if(self.termination.terminate(optimum_fitness, self._generations)):
                print "%i generations " % (self._generations)
                print "%i cfcs " % sum(self.logger.all()['count_cfc'])
                print "%i cfc spent for active planes" % self._spent
                print "%i ffcs " % sum(self.logger.all()['count_ffc'])
                break
        return self
