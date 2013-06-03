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

from pdb import set_trace
from numpy import matrix
from numpy.linalg import eigh, svd, norm

class ActivePlane(object):

    def __init__(self, nearest):
        """ nfeasibles and ninfeasibles are the nearest points
        approximated by e.g. LAHMCE e.g. (feasible, infeasible)"""

        means = []
        # calculate the means and calculate the distances
        # for uncertainty section
        for feasible, infeasible in nearest:
            means.append(0.5 * (feasible + infeasible))

        points = means

        self.centroid = sum(points) / len(points)
        rows = []
        for point in points:
            rows.append(point - self.centroid)
        M = matrix(rows)
        self.U, self.s, self.V = svd(M)

        sortkey = lambda t : t[0]
        self.normal = sorted(zip(self.s,self.V), key=sortkey)[0][1]

        # positive sign in feasible direction
        dec = (nearest[0][0] - self.centroid) * self.normal.T
        if dec < 0:
            self.normal = (-1) * self.normal

        # calculate uncertainty distance
        dists_feasible, dists_infeasible = [], []
        for feasible, infeasible in nearest:
            dec_f = (feasible - self.centroid) * self.normal.T
            dec_i = (infeasible - self.centroid) * self.normal.T
            dists_feasible.append(abs(dec_f))
            dists_infeasible.append(abs(dec_i))

        self.min_dist_feasible = min(dists_feasible)
        self.min_dist_infeasible = min(dists_infeasible)

        self.max_dist_feasible = max(dists_feasible)
        self.max_dist_infeasible = max(dists_infeasible)

    def predictable(self, x):
        dec = (x - self.centroid) * self.normal.T

        # if uncertain then raise exception
        if dec <= 0:
            if(abs(dec) < self.max_dist_infeasible):
                return False
            else:
                return True
        else:
            if(abs(dec) < self.max_dist_feasible):
                return False
            else:
                return True

    def predict(self, x):
        dec = (x - self.centroid) * self.normal.T

        # if uncertain then raise exception
        if dec <= 0:
            if(abs(dec) < self.max_dist_infeasible):
                raise Exception("Uncertain")
            else:
                return False
        else:
            if(abs(dec) < self.max_dist_feasible):
                raise Exception("Uncertain")
            else:
                return True

