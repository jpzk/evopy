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

class SchwefelsProblem240():

    description = "Schwefel's problem 2.40"
    description_short = "Schwefel240"

    def is_feasible(self, x):

        m = x
        # g1, g5       
        for i in range(0, x.size):
            if(m[0, i] < 0):
                return False
        # g6
        p = 50000
        sum_of_is = 0      
        for i in range(0, x.size):
            sum_of_is += (9+(i+1)) * m[0, i]
        left = -sum_of_is + p 
        if(left < 0):
            return False
        else: 
            return True

    def fitness(self, x):
        fitness = - x.sum()
        return fitness

    def optimum_fitness(self):
        return -5000
