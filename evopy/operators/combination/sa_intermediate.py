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

from evopy.individuals.selfadaptive_individual import SelfadaptiveIndividual

class SAIntermediate():
    def combine(self, pair):
        v1 = pair[0].value
        v2 = pair[1].value
        value = map(lambda i,j : (i+j)/2.0, v1, v2)
        sigma = ((pair[0].sigma + pair[1].sigma)/2.0)
        return SelfadaptiveIndividual(value, sigma)
       
