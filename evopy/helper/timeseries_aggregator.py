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
from numpy import sqrt, array
from copy import deepcopy

import pdb 

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
                    if(type(minimum_time_serie[k]) == type(None)):                    
                        minimum_time_serie[k] = time_series[k]
                        continue
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
                    if(type(maximum_time_serie[k]) == type(None)):
                        maximum_time_serie[k] = time_series[k]
                        continue           
                    if(time_series[k] > maximum_time_serie[k]):
                        maximum_time_serie[k] = time_series[k]

        return maximum_time_serie                        

    def get_aggregate(self):                     

        max_length = 0
        for time_serie in self._time_series:
            length = len(time_serie)
            if(length > max_length):
                max_length = length
        
        y = [[] for i in range(0, max_length)]
        for i in range(0, self._amount):
            time_series = self._time_series[i]
            for k in range(0, len(time_series)):
                if(type(time_series[k]) != type(None)):
                    y[k].append(time_series[k])
        
        y_means = map(lambda y_entry : array(y_entry).mean(), y)
        y_stds = map(lambda y_entry : array(y_entry).std(), y)
        y_vars = map(lambda y_entry : array(y_entry).var(), y)
        
        return y_means, y_stds 

