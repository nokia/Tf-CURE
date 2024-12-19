# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import functools
import inspect
from time import time
from contextlib import contextmanager
from typing import Callable, Dict, Any, Optional

@contextmanager
def timeit_cnt(name, *args, active: bool = True, **kwargs):

    start_time = time()
    yield
    delta_t = time() - start_time
    if active:
        print(f"{name} execution time: {float(delta_t*1000):.3f} ms")

class TimeitContext:
    def __init__(self):
        self.all_times = []
        self.execution_time = None
        self.n_runs = 0

    @contextmanager
    def __call__(self):
        start_time = time()  # Record the start time
        try:
            yield self  # Yield control to the block of code inside the context
        finally:
            end_time = time()  # Record the end time
            self.execution_time = end_time - start_time  # Calculate execution time
            self.all_times.append(self.execution_time)
            self.n_runs += 1

    @property
    def mean(self):
        return sum(self.all_times) / len(self.all_times)
