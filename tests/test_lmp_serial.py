#
# Copyright 2024-2025 Hannes Holey
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
import os
import pytest
from lammps import lammps


@pytest.fixture(scope='session')
def setup(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("tmp")
    logfile = os.path.join(tmpdir, "log.lammps")

    yield logfile


@pytest.mark.skip(reason="Skip tests that require LAMMPS for now")
def test_lmp_serial(setup):

    logfile = setup

    nargs = ["-screen", logfile, "-log", logfile]

    lmp = lammps(cmdargs=nargs)
    lmp.file(os.path.join('tests', 'in.lj'))
    lmp.close()

    with open(logfile, 'r') as f:
        last = f.readlines()[-1]

    assert last.startswith("Total wall time:")
