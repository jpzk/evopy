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

from evopy.helper.timeseries_aggregator import TimeseriesAggregator

def TA_get_minimum_test():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.2, 3.0, 6.0, 0.1, 0.6, 0.2, 0.6]
    c = [0.1, 0.1]

    minimum = TimeseriesAggregator([a,b,c]).get_minimum()
    if(minimum == [0.1, 0.1, 3.0, 0.1, 0.6, 0.2, 0.6]):
        assert True
    else:
        assert False

def TA_get_maximum_test():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.2, 3.0, 6.0, 0.1, 0.6, 0.2, 0.6]
    c = [0.1, 0.1]

    maximum = TimeseriesAggregator([a,b,c]).get_maximum()
    if(maximum == [1.0, 3.0, 6.0, 4.0, 5.0, 0.2, 0.6]):
        assert True
    else:
        assert False

