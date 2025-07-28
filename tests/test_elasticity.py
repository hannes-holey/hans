#
# Copyright 2020-2022, 2025 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Testing the elasticity wrapper class, not the computation of the elastic 
deformation itself. For the computation, refer to ContactMechanics.
"""

import pytest
import random
import numpy as np
from hans.elasticity import ElasticDeformation

@pytest.fixture
def setup():
    n = 20 + random.randint(1, 100)
    m = 20 + random.randint(1, 100)
    size = (n,m)
    dx = random.uniform(1e-5, 1e-1)
    dy = random.uniform(1e-5, 1e-1)
    E = 210e09
    v = random.uniform(0, 0.5)
    p = 10 ** (np.random.rand(*size)*8) # maximum: 1e09

    return size, dx, dy, E, v, p


periodicity_list = [(False, False), (True, False), (False, True), (True, True)]

# make sure that u is given as 2D numpy array with the correct dimensions for
# each periodicity and that it returns valid values
@pytest.mark.parametrize("periodic_x, periodic_y", periodicity_list)
def test_u_return_dimension(periodic_x, periodic_y, setup):
    size, dx, dy, E, v, p = setup
    ElDef = ElasticDeformation(size, dx, dy, E, v, periodic_x, periodic_y)
    u = ElDef.get_deformation(p)

    assert u.ndim == 2
    assert np.all(np.isfinite(u))
    assert u.shape == size
