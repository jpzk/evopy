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

    def savings(self):
        if(self.fp + self.tn == 0):
            return 0.0
        else:
            return self.tn / float(self.fp + self.tn)

    def acc(self):
        all_sum = self.matrix().sum()
        if(all_sum == 0):
            return 0.0
        else:
            return (self.tp + self.fp) / float(all_sum)

    def pacc(self):
        _sum = self.tp + self.fp
        if(_sum == 0):
            return 0.0
        return self.tp / _sum 

    def nacc(self):
        _sum = self.tn + self.fn
        if(_sum == 0):
            return 0.0
        return self.tn / _sum 

    def mcc(self):
        return self._mcc

    def matrix(self):
        return matrix([[self.tp, self.fp],[self.fn, self.tn]])
