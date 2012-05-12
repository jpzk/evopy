''' 
This file is part of evolutionary-algorithms-sandbox.

evolutionary-algorithms-sandbox is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evolutionary-algorithms-sandbox is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evolutionary-algorithms-sandbox.  If not, see <http://www.gnu.org/licenses/>.
'''

from sys import argv
from numpy import mean, var, std
from matplotlib.mlab import rec2txt, csv2rec, rec_groupby
from pylab import boxplot

data = csv2rec(argv[1], delimiter = ';')

def stats_for_attribute(attribute):
    statistics = rec_groupby(
        data, ['method'],
        [(attribute, min, 'min'),
        (attribute, mean, 'mean'),
        (attribute, max, 'max'),
        (attribute, var, 'var'),
        (attribute, std, 'std')])
    return rec2txt(statistics)

print 'constraint-calls'
print stats_for_attribute('constraintcalls')
print ''

print 'generations'
print stats_for_attribute('generations')
print ''

print 'sum-wrong-classification'
print stats_for_attribute('sumwrongclassification')
print ''
