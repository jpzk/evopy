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

from gtk import ComboBox
from gtk import ListStore
from gtk import CellRendererText

class ProblemComboBox(ComboBox):
    def __init__(self):
        super(ProblemComboBox, self).__init__()

        self.liststore = ListStore(str)
        self.set_model(self.liststore)
        self.cell = CellRendererText()
        self.pack_start(self.cell, True)
        self.add_attribute(self.cell, 'text', 0) 
        self.append_text("Tangent Restriction 2")
        self.set_active(0)
 
