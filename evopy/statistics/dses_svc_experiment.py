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

import csv

from evopy.examples import DSES
from evopy.examples import DSESSVC

file_call = 'experiment_calls.csv'
file_fitnesses = 'experiment_fitnesses.csv'
file_acc = 'experiment_acc.csv'

writer_calls = csv.writer(open(file_call, 'wb'), delimiter=';')
writer_fitnesses = csv.writer(open(file_fitnesses, 'wb'), delimiter=';')
writer_acc = csv.writer(open(file_acc, 'wb'), delimiter=';')

writer_calls.writerow(\
    ["sample", 
    "method",
    "train-function-calls",
    "constraint-calls",
    "metamodel-calls",
    "fitness-function-calls",
    "generations"])

writer_fitnesses.writerow(\
    ["sample", 
    "generation", 
    "worst-fitness", 
    "avg-fitness", 
    "best-fitness"])

writer_acc.writerow(\
    ["sample", 
    "method", 
    "generation", 
    "best-acc"])

# DSESSVC
for sample in range(0, 1):

    methodname = "DSESSVC"
    dsessvc = DSESSVC.get_method() 
    dsessvc.run()
    stats = dsessvc.get_statistics()

    writer_calls.writerow(\
        [sample, methodname,
        stats["train-function-calls"],
        stats["constraint-calls"],
        stats["metamodel-calls"],
        stats["fitness-function-calls"],
        stats["generations"]])

    best_fitnesses = stats["best-fitness"]
    worst_fitnesses = stats["worst-fitness"]
    avg_fitnesses = stats["avg-fitness"]

    for generation in range(0, int(stats["generations"])):
    
        worst_fitness = worst_fitnesses[generation]
        avg_fitness = avg_fitnesses[generation]
        best_fitness = best_fitnesses[generation]
        writer_fitnesses.writerow(\
            [sample, methodname, generation, worst_fitness, avg_fitness, best_fitness])
        writer_acc.writerow([sample, methodname, generation, stats["best-acc"][generation]])

for sample in range(0, 1):

    methodname = "DSES"
    dses = DSES.get_method() 
    dses.run()
    stats = dses.get_statistics()

    writer_calls.writerow(\
        [sample, methodname,
        0,
        stats["constraint-calls"],
        0,
        stats["fitness-function-calls"],
        stats["generations"]])

    best_fitnesses = stats["best-fitness"]
    worst_fitnesses = stats["worst-fitness"]
    avg_fitnesses = stats["avg-fitness"]

    for generation in range(0, int(stats["generations"])):
    
        worst_fitness = worst_fitnesses[generation]
        avg_fitness = avg_fitnesses[generation]
        best_fitness = best_fitnesses[generation]
        writer_fitnesses.writerow(\
            [sample, methodname, generation, worst_fitness, avg_fitness, best_fitness])

