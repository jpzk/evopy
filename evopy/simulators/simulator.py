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

from sys import path
path.append("../..")

# @todo spawn two processes and avoid playdoh map
multiprocessing = False

if(multiprocessing):
    from multiprocessing import cpu_count
    from evopy.external.playdoh import map

class Simulator():
    def __init__(self, optimizer, problem, accuracy):
        self.optimizer = optimizer
        self.problem = problem
        self.accuracy = accuracy 
	self.infeasibles = 0

    def simulate(self):
        while(True):

            # Simulator and optimizer handling constraints
            all_feasible = False
            while(not all_feasible):
                # ASK for solutions (feasbile and infeasible) 
                solutions = self.optimizer.ask_pending_solutions()

                # CHECK solutions for feasibility 
                feasibility =\
                    lambda solution : (solution, self.problem.is_feasible(solution))

                if(multiprocessing):
                    feasibility_information =\
                        map(feasibility, solutions, cpu = cpu_count())
                else:
                    feasibility_information =\
                        map(feasibility, solutions)

                # TELL feasibility, returns True if all feasible, 
                # returns False if extra checks
                all_feasible =\
                    self.optimizer.tell_feasibility(feasibility_information)

            # ASK for valid solutions (feasible)
            valid_solutions = self.optimizer.ask_valid_solutions()

            # CHECK fitness
            fitness = lambda solution : (solution, self.problem.fitness(solution))

            if(multiprocessing):
                fitnesses = map(fitness, valid_solutions, cpu = cpu_count())
            else:
                fitnesses = map(fitness, valid_solutions)

            # TELL fitness, return optimum
            optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)
            
            #dd = self.optimizer._meta_model.get_last_statistics()
            #mm_accuracy = mm_stats['best_acc']
            stats = self.optimizer.get_last_statistics()
            self.infeasibles += stats['infeasibles']
            
            print optimum_fitness, self.infeasibles

            # TERMINATION
            if(optimum_fitness <= self.problem.optimum_fitness() + self.accuracy):
                break

