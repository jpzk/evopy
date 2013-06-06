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

from playdoh import map as pmap
from itertools import product
from copy import deepcopy

class Sampling(object):

    def _run_value(self, value, parameter, args, algorithm):
        args[parameter] = value
        error = algorithm(args)
        algorithm = None
        return (value, error)

    def sample(self, parameters, intervals, stepfuncs, args, algorithm, parallel=False):

        steps = {}
        values = {}
        results = []

        for parameter, interval, stepfunc in zip(parameters, intervals, stepfuncs):
            # @todo if stepfunc function does not fit in interval.
            x = 0
            s = interval[0]
            y = interval[0]

            while(y < interval[1]):
                y = stepfunc(y)
                x += 1

            steps[parameter] = x
            values[parameter] = []

            values[parameter].append(s)
            y = s
            for step in xrange(steps[parameter]):
                y = stepfunc(y)
                values[parameter].append(y)

        iterables = []
        for k, vals in values.iteritems():
            iterables.append([(k, v) for v in vals])

        def run_with_parameter(combination):
            xargs = deepcopy(args)
            for parameter in combination:
                name, value = parameter
                xargs[name] = value
            return (xargs, algorithm(xargs))

        if parallel:
            combinations = [c for c in product(*iterables)]
            results = pmap(run_with_parameter, combinations)
        else:
            for combination in product(*iterables):
                results.append(run_with_parameter(combination))

        return results

