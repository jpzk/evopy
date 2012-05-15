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

import pylab
from numpy import unique
from matplotlib.mlab import rec2txt, csv2rec, rec_groupby

data = csv2rec('experiments/experiment_fitnesses.csv', delimiter=';')

fig = pylab.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('generations #')
ax1.set_ylabel('avg fitness')
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

pylab.title("[70;0.7]-DSES with SVC metamodel average fitness")
pylab.axis([0, 400, 2.0, 2.01])
generations = unique(data['generation'])

for sample in ["0","1"]:
    print sample
    sample = data[data['sample'] == int(sample)]
    avg_fitnesses = sample['avgfitness']
    best_fitnesses = sample['bestfitness']    
    worst_fitnesses = sample['worstfitness']    

    pylab.plot(generations, avg_fitnesses, "g")
#    pylab.plot(generations, best_fitnesses, "b")
#    pylab.plot(generations, worst_fitnesses, "r")



pylab.show()

