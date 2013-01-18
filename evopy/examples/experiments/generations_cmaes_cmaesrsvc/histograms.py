# Using the magic encoding
# -*- coding: utf-8 -*-

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
from sys import path
path.append("../../../..")

from copy import deepcopy
from numpy import matrix, log10, array, linspace, sqrt, pi, exp
from pickle import load 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import hist, plot
from setup import *

bff = file("output/generations.save", "r")
generations = load(bff)

def gauss(u):
    return (1.0 / sqrt(2 * pi)) * exp((-(1.0/2.0) * (u**2)))
 
def nadaraya(x, data, labels, h):
    labels = [0] + labels.tolist() + [0]   
    data = data.tolist()
    data = [data[0] - (data[1] - data[0])] + data 
    print data, labels
    bottom = sum(map(lambda sample : (1/h)*gauss((x - sample)/h), data))
    top = sum(map(lambda sample, label : label * (1/h)* gauss((x - sample)/h), data, labels))
    if(bottom == 0):
        return 0
    return float(top)/float(bottom)

for problem in problems:
    figure_hist = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
    logit = lambda value, optimum : log10(value - optimum)
    opt = problem().optimum_fitness()

    plt.xlabel('Genauigkeit in $\\log_{10}(f(\\vec{b}) - f(\\vec{x}^*))$')
    plt.ylabel('absolute H' + u'ä' + 'ufigkeit')

    x1 = map(lambda l : l[-1], generations[problem][optimizers[problem][0]])
    x2 = map(lambda l : l[-1], generations[problem][optimizers[problem][1]])

    x1_log = map(logit, x1, len(x1) * [opt])
    x2_log = map(logit, x2, len(x2) * [opt])

    minimum = min(x1_log + x2_log)
    maximum = max(x1_log + x2_log)

    plt.xlim([minimum - 2, maximum + 2])

    pdfs1, bins1, patches1 = hist(x1_log, normed=False, alpha=0.5,\
        histtype='step', edgecolor="g")

    h = 1.06 * array(x1_log).std() * (len(x1_log)**(-1.0/5.0))
    x = linspace(minimum - 2, maximum + 2, 100)
    y = map(lambda x : nadaraya(x, bins1, pdfs1, h), x)
    plot(x,y, linestyle="--", color="g")

    pdfs2, bins2, patches2 = hist(x2_log, normed=False, alpha=0.5,\
        histtype='step', edgecolor="#004779")

    h = 1.06 * array(x2_log).std() * (len(x2_log)**(-1.0/5.0))
    x = linspace(minimum - 2, maximum + 2, 100)
    y = map(lambda x : nadaraya(x, bins2, pdfs2, h), x)
    plot(x,y, linestyle="-", color="#004779")

    pp = PdfPages("output/%s.pdf" % str(problem).replace('.', '-'))
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.clf()


