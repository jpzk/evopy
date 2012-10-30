''' 
This file is part of evopy.

Copyright 2012, Jendrik Poloczek

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
import pdb
from numpy import vsplit

from sys import path
from evopy.helper.logger import Logger
path.append("../..")

class Simulator(object):
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

    def simulate(self):
        while(True):
            # Simulator and optimizer handling constraints
            all_feasible = False
            while(not all_feasible):
                # ASK for solutions (feasbile and infeasible) 
                solutions = self.optimizer.ask_pending_solutions()

                # CHECK solutions for feasibility 
                feasibility =\
                    lambda solution, position :\
                        (solution, self.problem.is_feasible(position))

                feasibility_information = []                   
                for solution in solutions:
                    self._count_cfc += 1                    
                    information = vsplit(solution, solution.shape[0])
                    position = information[0]
                    feasibility_information.append(feasibility(solution, position))
 
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
            self.logger.log()
            self._count_cfc = 0
            self._count_ffc = 0
          
            print "%.20f" % (optimum_fitness)

            # TERMINATION
            if(self.termination.terminate(optimum_fitness, self._generations)):
                break
            
        return self 
