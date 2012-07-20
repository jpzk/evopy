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

from random import sample

from evolution_strategy import EvolutionStrategy
from evopy.individuals.selfadaptive_individual import SelfadaptiveIndividual

class DPES(EvolutionStrategy): 

    description = "Death Penalty Evolution Strategy (DP-ES)"
    description_short = "DP-ES"

    def __init__(self, mu, lambd, sigmas):
        super(DPES, self).__init__(problem, mu, lambd) 
        
        self.current_population = []

        # valid solutions
        self._valid_solutions = []

        # statistics
        self._statistics_constraint_infeasibles_trajectory = []
        self._count_constraint_infeasibles = 0

    def ask_pending_solutions(self):
        """ ask pending solutions; solutions which need a checking for 
            true feasibility """        

        pending_solutions = []
        while(len(pending_solutions) < (self._lambd - len(self._valid_solutions))):

            # recombination
            father, mother = sample(self.current_population, 2)
            combination = lambda fv, mv : (fv + mv) / 2.0
            sigmas_combination = lambda fs, ms : (fs + ms) / 2.0
            new_value = map(combination, father.value, mother.value)
            new_sigmas = map(sigmas_combination, father.sigmas, mother.sigmas)
            child =  SelfadaptiveIndividual(new_value, new_sigmas)

            # mutation
            x = child.value
            new_x = map(lambda (xi, sigma): xi + gauss(0, sigma), zip(x, sigmas))
            child.value = new_x

            pending_solutions.append(child)
 
        return pending_solutions            

    def tell_feasibility(self, feasibility_information):
        """ tell feasibilty; return True if there are no pending solutions, 
            otherwise False """

         for (child, feasibility) in feasibility_information:
            if(feasibility):
                self._valid_solutions.append(child)
            else:
                self._count_constraint_infeasibles += 1

        # @todo shorten: return expression
        if(len(self._valid_solutions) < self._lambd):
            return False
        else:
            return True

    def ask_valid_solutions(self):
        return self._valid_solutions

    def tell_fitness(self, fitnesses):
        """ tell fitness; update all strategy specific attributes """        

        fitness = lambda (child, fitness) : fitness
        child = lambda (child, fitness) : child

        sorted_fitnesses = sorted(fitnesses, key = fitness)[:self._mu]
        self.current_population = map(child, sorted_fitnesses)

        ### UPDATE FOR NEXT ITERATION
        self._valid_solutions = []

        ### STATISTICS
        self._statistics_constraint_infeasibles_trajectory.append(\
            self._count_constraint_infeasibles)        
        self._count_constraint_infeasibles = 0                

        self._statistics_selected_children_trajectory.append(values)

        fitnesses = map(fitness, sorted_fitnesses)
        mean_fitness = array(fitnesses).mean()

        self._statistics_best_fitness_trajectory.append(best_fitness)
        self._statistics_worst_fitness_trajectory.append(worst_fitness)
        self._statistics_mean_fitness_trajectory.append(mean_fitness)

        # update best child, best fitness
        best_child, best_fitness = sorted_fitnesses[0]
        worst_child, worst_fitness = sorted_fitnesses[-1]        

        return best_child, best_fitness

    def get_statistics(self, only_last = False):
        select = lambda stats : stats[-1] if only_last else stats
        
        statistics = {}

        super_statistics = super(CMAES, self).get_statistics(only_last = only_last)
        for k in super_statistics:
            statistics[k] = super_statistics[k]

        return statistics

