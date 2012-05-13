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

from math import floor, sqrt
from collections import deque 

from svc_evolution_strategy import SVCEvolutionStrategy

class SVCCVDSBestSlidingWeighted(SVCEvolutionStrategy):
    """ Using the fittest feasible and infeasible individuals in a sliding
        window (between generations) to build a meta model using SVC. """
   
    _statistics_parameter_epsilon_trajectory = []
    
    def __init__(\
        self, problem, mu, lambd, alpha, sigma, theta, pi, epsilon, combination,\
        mutation, selection, view, beta, window_size, append_to_window,\
        crossvalidation, scaling, selfadaption):

        super(SVCCVDSBestSlidingWeighted, self).__init__(\
            problem, mu, lambd, alpha, sigma, combination, mutation,\
            selection, view, 0.0, 0.0)

        # Death Penalty step control parameters
        self._theta = theta
        self._pi = pi
        self._epsilon = epsilon

        self._selfadaption = selfadaption
        self._beta = beta
        self._append_to_window = append_to_window
        self._window_size = window_size
        self._crossvalidation = crossvalidation
        self._scaling = scaling            
        self._sliding_best_feasibles = deque(maxlen = self._window_size)
        self._sliding_best_infeasibles = deque(maxlen = self._window_size)

        # infeasibles
        self._DSES_infeasibles = 0

    def log(\
        self, generation, next_population, best_acc, parameter_C,\
        parameter_gamma, parameter_epsilon):
        
        super(SVCCVDSBestSlidingWeighted, self).log(\
            generation, next_population, best_acc, parameter_C,
            parameter_gamma)

        self._statistics_parameter_epsilon_trajectory.append(\
            parameter_epsilon)

    def view(\
        self, generation, next_population, best_acc, parameter_C,\
        parameter_gamma, parameter_epsilon):
        
        self._view.view(generation, next_population, self._problem.fitness,\
            best_acc, parameter_C, parameter_gamma, parameter_epsilon)

    def get_statistics(self):
        statistics = {
            "parameter-epsilon" : self._statistics_parameter_epsilon_trajectory}
        
        super_statistics = super(SVCCVDSBestSlidingWeighted, self).get_statistics()
        for k in super_statistics:
            statistics[k] = super_statistics[k]
        
        return statistics

    # generate child 
    def generate_child(self, population):
        tau = (1.0/sqrt(self._lambd)) 
        combined_child = self.combine(population)
        mutated_child = self.mutate(combined_child, combined_child.sigma)
        return self._selfadaption.mutate(mutated_child, tau)

    # main evolution 
    def _run(self, (population, generation, m, l, lastfitness,\
        alpha, epsilon)):
        """ This method is called every generation. """

        children = [self.generate_child(population) for child in range(0,l)]

        # Filter by checking feasiblity with SVC meta model, the 
        # meta model might be wrong, so we have to weighten between
        # the filtern with meta model and filtering with the 
        # true constraint function.
        cut = int(floor(self._beta * len(children)))
        meta_children = children[:cut]
        constraint_children = children[cut:]

        # check scaled against meta model, BUT the unscaled against 
        # the constraint function.
        meta_feasible_children = filter(\
            lambda child : self.is_meta_feasible(self._scaling.scale(child)), 
            meta_children)
        
        # Filter by true feasibility with constraind function, here we
        # can update the sliding feasibles and infeasibles.
        feasible_children = []
        infeasible_children = []
        
        for meta_feasible in meta_feasible_children: 
            if(self.is_feasible(meta_feasible)):
                feasible_children.append(meta_feasible)               
            else:
                infeasible_children.append(meta_feasible)
                self._DSES_infeasibles += 1
                # Because of Death Penalty we need a feasible reborn.
                reborn = []                
                while(len(reborn) < 1):  
                    generated = self.generate_child(population)
                    if(self.is_feasible(generated)):
                        reborn.append(generated)
                    else:
                        self._DSES_infeasibles += 1
                feasible_children.extend(reborn)                  

        # Filter the other part of the cut with the true constraint function. 
        # Using this information to update the meta model.
        for child in constraint_children:
            if(self.is_feasible(child)):
                feasible_children.append(child)
            else:
                infeasible_children.append(child) 
                self._DSES_infeasibles += 1
 
                # Because of Death Penalty we need a feasible reborn.
                reborn = []
                while(len(reborn) < 1):
                    generated = self.generate_child(population) 
                    if(self.is_feasible(generated)):
                        reborn.append(generated)
                    else:
                        self._DSES_infeasibles += 1
                feasible_children.extend(reborn)

        # feasible_children contains exactly lambda feasible children
        # and infeasible_children 
        next_population = self.select(population + feasible_children, m)

        map(self._sliding_best_infeasibles.append, 
            self.select(infeasible_children, self._append_to_window))

        map(self._sliding_best_feasibles.append,
            self.select(feasible_children, self._append_to_window))

        sliding_best_infeasibles =\
            [child for child in self._sliding_best_infeasibles]

        sliding_best_feasibles =\
            [child for child in self._sliding_best_feasibles]

        # new scaling because sliding windows changes
        self._scaling.setup(sliding_best_feasibles + sliding_best_infeasibles)

        scaled_best_feasibles = map(\
            self._scaling.scale, 
            self._sliding_best_feasibles)

        scaled_best_infeasibles = map(\
            self._scaling.scale,
            self._sliding_best_infeasibles)                

        best_parameters = self._crossvalidation.crossvalidate(\
            scaled_best_feasibles,
            scaled_best_infeasibles)

        self.train_metamodel(\
            feasible = best_parameters[0],
            infeasible = best_parameters[1],
            parameter_C = best_parameters[2],
            parameter_gamma = best_parameters[3])

        best_parameter_gamma = best_parameters[3]
        best_parameter_C = best_parameters[2]
        best_acc = best_parameters[4]

        fitness_of_best = self.fitness(next_population[0])

        self.log(generation, next_population, best_acc,\
            best_parameter_C, best_parameter_gamma, epsilon)

        self.view(generation, next_population, best_acc,\
            best_parameter_C, best_parameter_gamma, epsilon)

        if(self.termination(generation, fitness_of_best)):
            print next_population[0]
            return True
        else:
            return (next_population, generation + 1, m,\
            l, fitness_of_best, alpha, epsilon)

    def run(self):
        """ This method initializes the population etc. And starts the 
            recursion. """
        
        # check for feasiblity and initialize sliding feasible and 
        # infeasible populations.

        feasible_parents = []
        best_feasibles = []
        feasibles = []
        best_infeasibles = []
        infeasibles = []

        while(len(feasible_parents) < self._mu):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasible_parents.append(parent)
                feasibles.append(parent)
            else:
                infeasibles.append(parent)

        # just to be sure 
        while(len(infeasibles) < self._window_size):
            parent = self.generate_population()
            if(not self.is_feasible(parent)):
                infeasibles.append(parent)

        while(len(feasibles) < self._window_size):
            parent = self.generate_population() 
            if(self.is_feasible(parent)):
                feasibles.append(parent)

        # initial training of the meta model

        best_feasibles = self.select(feasibles, self._window_size)
        best_infeasibles = self.select(infeasibles, self._window_size)

        # adding to sliding windows

        map(self._sliding_best_feasibles.append, best_feasibles)
        map(self._sliding_best_infeasibles.append, best_infeasibles)

        # scaling, scaling factors are kept in scaling attribute.
        self._scaling.setup(best_feasibles + best_infeasibles)
        scaled_best_feasibles = self._scaling.scale(best_feasibles)
        scaled_best_infeasibles = self._scaling.scale(best_infeasibles)

        best_parameters = self._crossvalidation.crossvalidate(\
            scaled_best_feasibles,
            scaled_best_infeasibles)

        self.train_metamodel(\
            feasible = best_parameters[0],
            infeasible = best_parameters[1],
            parameter_C = best_parameters[2],
            parameter_gamma = best_parameters[3])

        result = self._run((feasible_parents, 0, self._mu, self._lambd,\
            0, self._alpha, self._sigma))

        while result != True:
            result = self._run(result)

        return result

"""
if __name__ == "__main__":
 
    sklearn_cv = SVCCVSkGrid(\
        gamma_range = [2 ** i for i in range(-15, 3, 2)],
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    method = SVCCVBestSlidingWeighted(\
        SphereProblem(),
        mu = 15,
        lambd = 100,
        alpha = 0.5,
        sigma = 1,
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        crossvalidation = sklearn_cv,
        scaling = ScalingDummy())
     
    method.run()
 """
