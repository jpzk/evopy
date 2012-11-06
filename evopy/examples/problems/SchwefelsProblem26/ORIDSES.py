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

from numpy import matrix
from evopy.strategies.ori_dses import ORIDSES 
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.simulators.simulator import Simulator
from evopy.operators.termination.accuracy import Accuracy

def get_method():
    method = ORIDSES(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0]])) 

    return method

if __name__ == "__main__":
    problem = SchwefelsProblem26()
    termination = Accuracy(problem.optimum_fitness(), pow(10, -6))
    sim = Simulator(get_method(), problem, termination)
    results = sim.simulate()
