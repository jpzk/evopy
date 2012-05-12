''' 
This file is part of evolutionary-algorithms-sandbox.

evolutionary-algorithms-sandbox is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evolutionary-algorithms-sandbox is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evolutionary-algorithms-sandbox.  If not, see <http://www.gnu.org/licenses/>.
'''

from numpy import array
from scaling_strategy import ScalingStrategy

class ScalingStandardscore(ScalingStrategy):
    """ Scaling to standardscore """

    def setup(self, individuals):
        values = map(lambda i : i.value, individuals)
        iarray = array(values)
        self._mean = iarray.mean()
        self._std = iarray.std()

    def scale(self, individuals):
        values = map(lambda i : i.value, individuals)
        return map(\
            lambda x : (x - self._mean) / self._std, 
            values)

