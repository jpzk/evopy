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

from math import floor

class DSESView():
    def view(\
        self, generations, next_population, fitness, epsilon,\
        DSES_infeasibles, sigmasmean):

        population = sorted(next_population, key=lambda child : fitness(child))
        best_fitness = fitness(population[0])

        print("gen %i - fit: %f - e: %f - inf: %i - sm: %f " %\
            (generations, best_fitness, epsilon, DSES_infeasibles,\
            sigmasmean))
       
