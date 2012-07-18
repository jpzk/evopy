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

class SimulationStatistics():
    def __init__(self, opt, problem, acc, sim_stats, opt_stats,\
        mm_stats = False):

        self.optimizer_name = opt
        self.problem_name = problem
        self.accuracy = acc
        self.simulator = sim_stats 
        self.optimizer = opt_stats

        if(mm_stats != False):
            self.metamodel = mm_stats
        else:
            self.metamodel = self.dummy_mm_statistics(1)

    def dummy_mm_statistics(self, generations):
        return {\
            "angles" : [0] * generations,\
            "best_parameter_C" : [0] * generations,\
            "best_acc": [0] * generations}

    def cumulated_cfc(self):
        cumulated_cfc = 0
        for cfc in self.simulator['cfc']:
            cumulated_cfc += int(cfc)
        return cumulated_cfc      


