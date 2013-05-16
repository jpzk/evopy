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

from numpy import matrix, sqrt

class ConfusionMatrix():

    def __init__(self, apos_feasibility):
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0

        for solutions, meta_feasibility, true_feasibility in apos_feasibility:
            if(meta_feasibility and true_feasibility):
                self.tp += 1.0
            elif(meta_feasibility and not true_feasibility):
                self.fp += 1.0
            elif(not meta_feasibility and true_feasibility):
                self.fn += 1.0
            else:
                self.tn += 1.0

        top = self.tp * self.tn - self.fp * self.fn
        bottom = sqrt((self.tp + self.fp) * (self.tp + self.fn) *\
            (self.tn + self.fp) * (self.tn + self.fn))
        if(bottom == 0):
            self._mcc = top / 1.0
        else:
            self._mcc = top / bottom

    def success_probability(self):
        sum_b = self.tn + self.fn + self.tp + self.fp
        sum_t = self.tp + self.fn

        if(sum_b == 0):
            return 0.0
        else:
            return sum_t / float(sum_b)

    # negative prediction value
    def npv(self):
        sum_ = self.tn + self.fn
        if(sum_ == 0):
            return 0.0
        else:
            return self.tn / float(sum_)

    # positive prediction value
    def ppv(self):
        if(self.tp + self.fp == 0):
            return 0.0
        else:
            return self.tp / float(self.tp + self.fp)

    def matrix(self):
        return matrix([[self.tp, self.fp],[self.fn, self.tn]])
