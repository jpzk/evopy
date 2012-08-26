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

import numpy
import pickle
import marshal
import types
import binascii
from sys import stdin, stdout, exc_info

def coroutine(func):
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.next()
        return cr
    return start        

@coroutine
def fitnesscr():
    fitness_func = None
    recv = (yield)
    code = marshal.loads(recv)            
    fitness_func = types.FunctionType(code, globals(), "name")
    stdout.write("configured" + "\n") 
    stdout.flush()

    while True:
        recv = (yield)
        fitness = fitness_func(numpy.loads(binascii.a2b_hex(recv[:-1])))
        stdout.write(binascii.b2a_hex(numpy.array(fitness).dumps()) + "\n")
        stdout.flush()

cr = fitnesscr()
while True:
    piped = stdin.readline()
    cr.send(piped)
    
