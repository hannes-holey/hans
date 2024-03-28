import os
import sys
import subprocess
import pytest


@pytest.fixture(scope="session", params=[1, 2, 3, 4, 5, 6, 7, 8])
def command(request):

    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spawner.py')
    command = f'mpirun -n 1 --oversubscribe {sys.executable} {fname} {request.param}'

    yield command


def test_spawn(command):

    try:
        subprocess.run(command.split(), check=True, stderr=True, env=os.environ)
    except subprocess.CalledProcessError:
        assert False
