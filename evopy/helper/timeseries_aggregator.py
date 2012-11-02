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
from numpy import sqrt
from copy import deepcopy

class TimeseriesAggregator():
    def __init__(self, time_series):
        self._time_series = time_series
        self._amount = len(time_series)

    def get_minimum(self):
        max_length = 0        
        
        for time_serie in self._time_series: 
            length = len(time_serie)
            if(length > max_length):
                max_length = length
                longest = time_serie        

        minimum_time_serie = deepcopy(longest)

        for i in range(0, self._amount):
            time_series = self._time_series[i]
            for k in range(0, len(time_series)):
                if(type(time_series[k]) != type(None)):
                    if(time_series[k] < minimum_time_serie[k]):
                        minimum_time_serie[k] = time_series[k]

        return minimum_time_serie                        
   
    def get_maximum(self):
        max_length = 0        
        
        for time_serie in self._time_series: 
            length = len(time_serie)
            if(length > max_length):
                max_length = length
                longest = time_serie        

        maximum_time_serie = deepcopy(longest)

        for i in range(0, self._amount):
            time_series = self._time_series[i]
            for k in range(0, len(time_series)):
                if(type(time_series[k]) != type(None)):
                    if(time_series[k] > maximum_time_serie[k]):
                        maximum_time_serie[k] = time_series[k]

        return maximum_time_serie                        

    def get_aggregate(self):
        max_length = 0
        for time_serie in self._time_series:
            length = len(time_serie)
            if(length > max_length):
                max_length = length

        std_time_series = [0 for i in range(0, max_length)]
        sum_time_series = [0 for i in range(0, max_length)] 
        amount_time_series = [0 for i in range(0, max_length)]
        mean_time_series = [0 for i in range(0, max_length)]

        for i in range(0, self._amount):
            time_series = self._time_series[i]
            for k in range(0, len(time_series)):
                if(type(time_series[k]) != type(None)):
                    amount_time_series[k] += 1
                    sum_time_series[k] += time_series[k]

        for i in range(0, max_length):
            sums = sum_time_series[i]
            amount = amount_time_series[i]
            if(amount >= 1):
                mean_time_series[i] = float(sums) / float(amount)
            else:
                mean_time_series[i] = 0

        variance_sums = [0 for i in range(0, max_length)]
        for i in range(0, max_length):
            for k in range(0, self._amount):
                if(len(self._time_series[k]) > i):
                    if(type(self._time_series[k][i]) != type(None)):
                        variance_sums[i] +=\
                            (self._time_series[k][i] - mean_time_series[i]) ** 2
        
        variance_avgs = map(lambda x : x / float(len(variance_sums)), variance_sums)
        stds = map(lambda x : sqrt(x), variance_avgs)  

        return mean_time_series, stds

