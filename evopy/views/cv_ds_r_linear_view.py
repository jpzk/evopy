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
from view import View

class CVDSRLinearView(View):

    def view(\
        self, generations, next_population, fitness, best_acc,\
        parameter_C, epsilon, DSES_infeasibles, wrong_meta_feasibles, sigmasmean):

        population = sorted(next_population, key=lambda child : fitness(child))
        best_fitness = fitness(population[0])
        best_acc = floor(100 * best_acc)

        output = ("gen %i - fit: %f " + 
                "C: %.2f - acc: %.2f - e: %f - inf: %i/%i - sm: %f ") %\
                (generations, best_fitness, parameter_C,\
                best_acc, epsilon, wrong_meta_feasibles, DSES_infeasibles,\
                sigmasmean)

        self._output(output)