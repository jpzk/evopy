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

import csv
from copy import deepcopy
from numpy import matrix, log10, array
from scipy.stats import wilcoxon 
from scipy import polyfit, polyval 
from itertools import chain
from pylab import plot 
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

f = open("output/cpus2.csv")
reader = csv.reader(f)

cpus_ids = range(0,8)
cpus = {}
for i in cpus_ids:
    cpus[i] = [] 

indices = [0,6,12,18,24,30,36,42]
for row in reader:
    for cpu in cpus_ids:
        cpus[cpu].append(row[indices[cpu]])

means,stds = {}, {}
for i in cpus_ids:
    means[i] = 0
    stds[i] = 0

for cpu in cpus_ids:
    vals = array(map(lambda s : float(s), cpus[cpu][:-87]))
    means[cpu] = vals.mean()
    stds[cpu] = vals.std()

results = file("output/load.tex", "w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X X X X X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Kennzahl} & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\\\\n",
    "\midrule\n"]

endlines = [
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

means_str = "Mittelwert"
for cpu in cpus_ids:
    means_str += "& %1.2f " % means[cpu] 
means_str += "\\\\\n"    

stds_str = "Standardabweichung"
for cpu in cpus_ids:
    stds_str += "& %1.2f " % stds[cpu]
stds_str += "\\\\\n"    

lines = lines + [means_str] + [stds_str] + endlines
results.writelines(lines)
results.close()



    
        
