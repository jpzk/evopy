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

from multiprocessing import cpu_count
from playdoh import map
from sys import stdout
from math import floor
from os import makedirs
from csv import writer

def call_case(case):
    result = case()
    return (result, result.get_statistics())

class Experiment(object):

    parent_dir = "evopy_experiments/"
    f_call = "/calls.csv"
    f_fitness = "/fitness.csv"
    f_acc = "/accuracy.csv"

    general_attributes =\
        {"problem" : "problem", 
        "method" : "method",
        "sample" : "sample"}

    fitness_attributes =\
        {"generation" : "generation",
        "best-fitness" : "best-fitness",
        "avg-fitness" : "avg-fitness",
        "worst-fitness" : "worst-fitness"}

    calls_attributes =\
        {"train-function-calls" : "train-function-calls" ,\
        "constraint-calls" : "constraint-calls",\
        "metamodel-calls" : "metamodel-calls",\
        "fitness-function-calls" : "fitness-function-calls",\
        "generations" : "generations"}

    accuracy_attributes =\
        {"generation" : "generation", 
        "best-acc" : "best-acc"}            

    def __init__(self, problem, directory):

        folder = self.parent_dir + directory
        makedirs(folder)

        self._writer_calls = writer(open(folder + self.f_call, 'wb'), delimiter=';')
        self._writer_fitnesses = writer(open(folder + self.f_fitness, 'wb'), \
            delimiter=';')

        self._writer_acc = writer(open(folder + self.f_acc, 'wb'), delimiter=';')
        self._problem = problem

        csv_calls_attributes = self.general_attributes.keys() +\
            self.calls_attributes.keys()
        csv_fitness_attributes = self.general_attributes.keys() +\
            self.fitness_attributes.keys()
        csv_accuracy_attributes = self.general_attributes.keys() +\
            self.accuracy_attributes.keys()

        self._writer_calls.writerow(csv_calls_attributes)
        self._writer_fitnesses.writerow(csv_fitness_attributes)
        self._writer_acc.writerow(csv_accuracy_attributes)

    def run_cases(self, cases, samples):
        for case in cases:
            results = map(call_case, [case] * samples, cpu = cpu_count())
            for sample, result in enumerate(results):
                obj, statistics = result
                print statistics
                method = obj._strategy_name
                meta_stats =\
                    {"problem" : self._problem,\
                    "method" : method,\
                    "sample" : sample}
                self._write_stats(meta_stats, statistics)
                self._update_progress(sample + 1, samples, method)               

    def _write_stats(self, meta_stats, stats):
        '''no documentation yet'''

        generations = int(stats[self.calls_attributes["generations"]])
        problem = meta_stats[self.general_attributes["problem"]]
        method = meta_stats[self.general_attributes["method"]]
        sample = meta_stats[self.general_attributes["sample"]]

        # general cases
        constraint_calls = stats[self.calls_attributes["constraint-calls"]]
        fitness_calls = stats[self.calls_attributes["fitness-function-calls"]]
        generations = stats[self.calls_attributes["generations"]]

        # special cases
        if(self.calls_attributes["train-function-calls"] in stats.keys()):
            train_calls =\
                stats[self.calls_attributes["train-function-calls"]]
        else:
            train_calls = 0

        if(self.calls_attributes["metamodel-calls"] in stats.keys()):
            metamodel_calls =\
                stats[self.calls_attributes["metamodel-calls"]]
        else:
            metamodel_calls = 0

        self._writer_calls.writerow(\
            [problem, 
            method, 
            sample,
            train_calls,
            constraint_calls,
            metamodel_calls,
            fitness_calls,
            generations])

        best_fitnesses = stats[self.fitness_attributes["best-fitness"]]
        worst_fitnesses = stats[self.fitness_attributes["worst-fitness"]]
        avg_fitnesses = stats[self.fitness_attributes["avg-fitness"]]

        # special case
        if(self.accuracy_attributes["best-acc"] in stats.keys()):
            best_acc = stats[self.accuracy_attributes["best-acc"]]
        else: 
            best_acc = [0 for i in range(0, generations)]

        for generation in range(0, generations):
            
            worst_fitness = worst_fitnesses[generation]
            avg_fitness = avg_fitnesses[generation]
            best_fitness = best_fitnesses[generation]

            self._writer_fitnesses.writerow([problem, method, sample,\
                generation, worst_fitness, avg_fitness, best_fitness])

            self._writer_acc.writerow([problem, method, sample, \
                generation, best_acc[generation]])

    def _done(self, i, n, msg):
        s = "["
        percentage = int(floor((float(i)/float(n)*10)))
        for i in range(0, percentage):
            s += "="
        for i in range(percentage, 10):
            s += " "
        s += "] " + msg
        return s

    def _update_progress(self, i, n, msg):
        stdout.write('\r'*(12+len(msg)))
        stdout.flush()
        msg = self._done(i, n, msg)      
        stdout.write(msg)
        stdout.flush()


