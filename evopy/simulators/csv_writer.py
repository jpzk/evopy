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
import pdb
from os import makedirs
from csv import writer

parent_dir = "evopy_experiments/"

class CSVWriter():

    general_cols = ["optimizer", "problem", "accuracy"]

    def __init__(self, directory):
        self.path = parent_dir + directory
        makedirs(self.path)
        
        self._init_cfc()
        self.write_funcs = [self._write_cfc]

    def _init_cfc(self):
        cfc_csv = "/cfc.csv"
        col_name = "cfc"
        self.cfc_writer = writer(open(self.path + cfc_csv, "wb"), delimiter=';')
        col_names = self.general_cols 
        col_names.append(col_name)
        self.cfc_writer.writerow(col_names)

    def _general_cols(self, simulation_statistics):
        return [simulation_statistics.optimizer_name,\
            simulation_statistics.problem_name,\
            simulation_statistics.accuracy]

    def _write_cfc(self, simulation_statistics):
        gcols = self._general_cols(simulation_statistics)
        cols = gcols
        cols.append(str(simulation_statistics.cumulated_cfc()))
        self.cfc_writer.writerow(cols)
        
    def write(self, simulation_statistics):
        for write_func in self.write_funcs:
            write_func(simulation_statistics)
