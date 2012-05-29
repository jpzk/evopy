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

class View(object):

    def __init__(self, mute = False, delegate_output = False):
        self._mute = mute
        self._delegate_output = delegate_output

    def _output(self, output):
        if not self._mute:
            if not self._delegate_output:
                print output
            else: 
                self._delegate_output.delegate(output)

