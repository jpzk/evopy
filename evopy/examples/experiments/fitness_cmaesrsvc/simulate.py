''' 
This file is part of evopy.

Copyright 2012 - 2013, Jendrik Poloczek

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
path.append("../../../..")

from pickle import dump
from copy import deepcopy
from numpy import matrix, log10

from evopy.simulators.simulator import Simulator
from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 
from evopy.external.playdoh import map as pmap

from os.path import exists
from os import mkdir

from setup import *  

# create simulators
for problem in problems:
    for optimizer in optimizers[problem]:
        simulators_op = []
        for i in range(0, samples):
            simulator = Simulator(optimizer(), problem(), termination)
            simulators_op.append(simulator)
        simulators[problem][optimizer] = simulators_op

simulate = lambda simulator : simulator.simulate()

# run simulators 
for problem in problems:
    for optimizer, simulators_ in simulators[problem].iteritems():
        resulting_simulators = pmap(simulate, simulators_)
        for simulator in resulting_simulators:
            fitness = simulator.optimizer.logger.all()['best_fitness']
            fitnesses[problem][optimizer].append(fitness)

if not exists("output/"): 
    mkdir("output/")

fitnesses_file = open("output/fitnesses_file.save", "w")
dump(fitnesses, fitnesses_file)
fitnesses_file.close()
