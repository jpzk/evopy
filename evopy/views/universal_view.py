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

class UniversalView(View):

    def view(\
        self, generations = None, best_fitness = None, best_acc = None,\
        parameter_C = None, epsilon = None, DSES_infeasibles = None,\
        wrong_meta_infeasibles = None, angles = None, sigmasmean = None):

        line = ""

        if(generations != None):
            line += "gen: %i " % (generations) 
        if(best_fitness != None):
            line += "bf: %f " % (best_fitness) 
        if(best_acc != None):
            line += "ba: %f " % (best_acc) 
        if(parameter_C != None):
            line += "C: %f " % (parameter_C) 
        if(epsilon != None):
            line += "e: %f " % (epsilon) 
        if(DSES_infeasibles != None):
            line += "d-inf: %i " % (DSES_infeasibles) 
        if(wrong_meta_infeasibles != None):
            line += "m-inf: %i " % (wrong_meta_infeasibles) 
                    
        if(angles != None):
            line += "\nangles: "
            for angle in angles:
                line += "%f " % (angle) 
           
        self._output(line)
