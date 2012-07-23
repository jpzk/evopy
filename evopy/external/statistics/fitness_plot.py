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
from sys import argv
from numpy import unique
from matplotlib.mlab import rec2txt, csv2rec, rec_groupby

data = csv2rec(argv[1], delimiter=';')

for sample in [0]:
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('generations #')
    ax1.set_ylabel('avg fitness')
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    sample = data[data['sample'] == sample]

    pylab.title(argv[2])
    pylab.axis([0, sample['generation'].max(), 25.0, 100.0])

    generations = sample['generation']
    avg_fitnesses = sample['avgfitness']
    best_fitnesses = sample['bestfitness']    
    worst_fitnesses = sample['worstfitness']    

#    pylab.plot(generations, best_fitnesses, "r",antialiased=True )  
    pylab.plot(generations, avg_fitnesses, "g", antialiased=True)    
#    pylab.plot(generations, worst_fitnesses, "b")    
   
pylab.show()

