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

class ORCombinator():
    def __init__(self, conditions):
        self._conditions = conditions 

    def terminate(self, best_fitness, generations):
        delegate = lambda c : c.terminate(best_fitness, generations)
        results = map(delegate, self._conditions)
        return reduce(lambda c1, c2 : c1 or c2, results)
        
