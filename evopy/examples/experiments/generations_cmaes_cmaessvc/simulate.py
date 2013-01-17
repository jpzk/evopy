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
            opt_fit = problem().optimum_fitness()
            termination = Accuracy(opt_fit, accuracy)
            simulator = Simulator(optimizer(), problem(), termination)
            simulators_op.append(simulator)
        simulators[problem][optimizer] = simulators_op

simulate = lambda simulator : simulator.simulate()

# run simulators 
for problem in problems:
    for optimizer, simulators_ in simulators[problem].iteritems():
        resulting_simulators = pmap(simulate, simulators_)
        for simulator in resulting_simulators:
            generation = simulator.logger.all()['generations']
            generations[problem][optimizer].append(generation)

if not exists("output/"): 
    mkdir("output/")

bf_file = open("output/generations_file.save", "w")
dump(generations, bf_file)
bf_file.close()
