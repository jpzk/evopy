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
from evopy.individuals.selfadaptive_individual import SelfadaptiveIndividual

class SAIntermediate():
    def combine(self, population):
        father, mother = sample(population, 2)
        combination = lambda fv, mv : (fv + mv) / 2.0
        sigmas_combination = lambda fs, ms : (fs + ms) / 2.0
        new_value = map(combination, father.value, mother.value)
        new_sigmas = map(sigmas_combination, father.sigmas, mother.sigmas)
        return SelfadaptiveIndividual(new_value, new_sigmas)

