# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
#
# Contact: Mattia Milani (Nokia) <mattia.milani@nokia.com>

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
