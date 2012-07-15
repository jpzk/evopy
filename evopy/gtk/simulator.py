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

import time
import threading

class Simulator(threading.Thread):
    """ This class represents the simulator thread """           

    def configure(self, optimizer, problem, accuracy):
        self.optimizer = optimizer
        self.problem = problem
        self.accuracy = accuracy

    def run(self):                  
        self.gui_closed = False
        self.stop = False

        while(not self.gui_closed and not self.stop):
            # Simulator and optimizer handling constraints
            all_feasible = False
            while(not all_feasible):
                # ASK for solutions (feasbile and infeasible) 
                solutions = self.optimizer.ask_pending_solutions()

                # CHECK solutions for feasibility 
                feasibility =\
                    lambda solution : (solution, self.problem.is_feasible(solution))

                feasibility_information = map(feasibility, solutions)

                # TELL feasibility, returns True if all feasible, 
                # returns False if extra checks
                all_feasible =\
                    self.optimizer.tell_feasibility(feasibility_information)

            # ASK for valid solutions (feasible)
            valid_solutions = self.optimizer.ask_valid_solutions()

            # CHECK fitness
            fitness =\
                lambda solution : (solution, self.problem.fitness(solution))
            fitnesses = map(fitness, valid_solutions)

            # TELL fitness, return optimum
            optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)

            # GUI update
            optimizer_stats = self.optimizer.get_last_statistics()
            metamodel_stats = self.optimizer._meta_model.get_last_statistics()
            self.gui.on_update_plots(optimizer_stats, metamodel_stats)
                                  
            time.sleep(0.5)

            # TERMINATION
            if(optimum_fitness <= self.problem.optimum_fitness() + self.accuracy):
                break

