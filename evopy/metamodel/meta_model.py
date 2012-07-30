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

# note: Logger must not be an inner class, errors with playdoh. 
class Logger(object):
    def __init__(self, scope):
        self.logs, self.bindings, self.const_bindings = {}, {}, {}
        self.scope = scope

    def add_const_binding(self, var_name, name):
        self.const_bindings[name] = var_name       

    def add_binding(self, var_name, name):
        self.bindings[name] = var_name
        self.logs[name] = []

    def const_log(self):
        for k, v in self.const_bindings.iteritems():
            self.logs[k] = self.scope.__getattribute__(v)

    def log(self):
        for k, v in self.bindings.iteritems():
            self.logs[k].append(self.scope.__getattribute__(v))                

class MetaModel(object):
    def __init__(self):
        self.logger = Logger(self)
