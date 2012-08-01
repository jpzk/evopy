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

from evopy.simulators.simulation_statistics import SimulationStatistics

class ExperimentSimulator():

    def __init__(self, optimizer, problem, accuracy):
        self.optimizer = optimizer
        self.problem = problem
        self.accuracy = accuracy
        self._count_cfc = 0
        self._statistics_cfc_trajectory = []
        self.infeasibles = 0

        self._general_optimizer = optimizer.description_short       
        self._general_problem = problem.description_short
        self._general_accuracy = str(accuracy)

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

                feasibility_information = []                   
                for solution in solutions:
                    self._count_cfc += 1 
                    feasibility_information.append(feasibility(solution))
 
                # TELL feasibility, returns True if all feasible, 
                # returns False if extra checks
                all_feasible =\
                    self.optimizer.tell_feasibility(feasibility_information)

            # ASK for valid solutions (feasible)
            valid_solutions = self.optimizer.ask_valid_solutions()

            # CHECK fitness
            fitness = lambda solution : (solution, self.problem.fitness(solution))
            fitnesses = map(fitness, valid_solutions)

            # TELL fitness, return optimum
            optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)
 
            # A-POSTERIORI information for confusion matrix
            if('ask_a_posteriori_solutions' in dir(self.optimizer)):
                apos_feasibility =\
                    lambda (solution, meta_feasibility) :\
                    (solution, meta_feasibility, self.problem.is_feasible(solution))

                apos_solutions = self.optimizer.ask_a_posteriori_solutions()
                feasibility_info = map(apos_feasibility, apos_solutions)
                self.optimizer.tell_a_posteriori_feasibility(feasibility_info)

            # UPDATE OWN STATS                                   
            self._statistics_cfc_trajectory.append(self._count_cfc)
            self._count_cfc = 0
           
            print optimum_fitness

            # TERMINATION
            if(optimum_fitness <= self.problem.optimum_fitness() + self.accuracy):
                break
       
        # CREATE SimulationStatistics object
        if('_meta_model' in dir(self.optimizer)):
            simulation_statistics = SimulationStatistics(\
                self._general_optimizer,\
                self._general_problem,\
                self._general_accuracy,\
                self.get_statistics(),\
                self.optimizer.get_statistics(),\
                mm_stats = self.optimizer._meta_model.get_statistics())
        else:                
            simulation_statistics = SimulationStatistics(\
                self._general_optimizer,\
                self._general_problem,\
                self._general_accuracy,\
                self.get_statistics(),\
                self.optimizer.get_statistics())

        return simulation_statistics

    def get_statistics(self):
        statistics = {
            "cfc" : self._statistics_cfc_trajectory}
        return statistics

    def get_last_statistics(self):
        statistics = {
            "cfc" : self._statistics_cfc_trajectory[-1]}
        return statistics
