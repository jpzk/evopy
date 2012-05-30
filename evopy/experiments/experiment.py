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

from sys import stdout
from math import floor

class Experiment():
    def done(self, i, n, msg):
        s = "["
        percentage = int(floor((float(i)/float(n)*10)))
        for i in range(0, percentage):
            s += "="
        for i in range(percentage, 10):
            s += " "
        s += "] " + msg
        return s

    def update_progress(self, i, n, msg):
        stdout.write('\r'*(12+len(msg)))
        stdout.flush()
        msg = self.done(i, n, msg)      
        stdout.write(msg)
        stdout.flush()


