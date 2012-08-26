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

import numpy
import binascii
import marshal
from threading import Thread
from Queue import Queue
import subprocess
from sys import path
path.append("../..")

class MultiprocessSimulator():
    """ Actor-based multiprocessing simulator """                

    def __init__(self, optimizer, problem, accuracy):
        self.optimizer = optimizer
        self.problem = problem
        self.accuracy = accuracy
        self._count_cfc = 0
        self._statistics_cfc_trajectory = []
        self.infeasibles = 0

        self._processes = 1 
        self._evaluators = []
        self._queue = []

        for i in range(0, self._processes):
            e = actor_fitness(self._queue)
            e.next()
            e.send((MSG_FUNCTION_CODE, self.problem.fitness))
            self._evaluators.append(e)

    def simulate(self):
        while(True):
            # Simulator and optimizer handling constraints
            all_feasible = False
            while(not all_feasible):
                # ASK for solutions (feasbile and infeasible) 
                solutions = self.optimizer.ask_pending_solutions()

                # CHECK solutions for feasibility 
                feasibility =\
                    lambda solution : (solution, self.problem.is_feasible(solution))

                feasibility_information = []                   
                for solution in solutions:
                    self._count_cfc += 1 
                    feasibility_information.append(feasibility(solution))
 
                # TELL feasibility, returns True if all feasible, 
                # returns False if extra checks
                all_feasible =\
                    self.optimizer.tell_feasibility(feasibility_information)

            # ASK for valid solutions (feasible)
            valid_solutions = self.optimizer.ask_valid_solutions()            
            
            map(lambda solution : self._evaluators[0].send((MSG_PROCESS, numpy.array(solution.value))),\
                valid_solutions)
            
            while(self._queue < len(valid_solutions)):
                print ".",
            
            # CHECK fitness
            #fitness = lambda solution : (solution, self.problem.fitness(solution))
            #fitnesses = map(fitness, valid_solutions)
            fitnesses = self._queue 
            self._queue = []

            # TELL fitness, return optimum
            optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)

            # A-POSTERIORI information for confusion matrix
            if('ask_a_posteriori_solutions' in dir(self.optimizer)):
                apos_feasibility =\
                    lambda (solution, meta_feasibility) :\
                    (solution, meta_feasibility, self.problem.is_feasible(solution))

                apos_solutions = self.optimizer.ask_a_posteriori_solutions()
                feasibility_info = map(apos_feasibility, apos_solutions)
                self.optimizer.tell_a_posteriori_feasibility(feasibility_info)

            # UPDATE OWN STATS                                   
            self._statistics_cfc_trajectory.append(self._count_cfc)
            self._count_cfc = 0
          
            print "%.20f %i" % (optimum_fitness, self._statistics_cfc_trajectory[-1])

            # TERMINATION
            if(optimum_fitness <= self.problem.optimum_fitness() + self.accuracy):
                print sum(self._statistics_cfc_trajectory)
                break

    def get_statistics(self, only_last = False):
        select = lambda stats : stats[-1] if only_last else stats

        statistics = {"cfc" : select(self._statistics_cfc_trajectory)}
        return statistics

MSG_FUNCTION_CODE = 0
MSG_PROCESS = 1 
MSG_TERMINATE = 2

def actor_fitness(response_queue):
    spawn = "multiprocess_spawn.py"
    pipe = subprocess.Popen(['python', spawn],\
        stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    messages = Queue()

    def run_target():
        try:
            while True:                
                message = messages.get()
                
                # terminate-message
                if(message == MSG_TERMINATE):
                    pipe.kill()
                    break
                else:
                    command, data = message

                # upload fitness function code
                if(command == MSG_FUNCTION_CODE):
                    func_code = marshal.dumps(data.func_code)
                    pipe.stdin.write(func_code + "\n")
                    pipe.stdin.flush()
                    output = pipe.stdout.readline()

                # process an individual
                elif(command == MSG_PROCESS):
                    send = binascii.b2a_hex(data.dumps()) + "\n"
                    pipe.stdin.write(send)
                    pipe.stdin.flush()
                    output = binascii.a2b_hex(pipe.stdout.readline()[:-1])
                    response_queue.append((data, numpy.loads(output)))
        except GeneratorExit:                       
            pipe.kill()

    Thread(target=run_target).start()
    try: 
        while True: 
            child = (yield)
            print child
            messages.put(child)
    except GeneratorExit:
        messages.put(MSG_TERMINATE)

       
