import os
import subprocess


def test_spawn():

    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spawner.py')
    command = f'mpirun -n 1 --oversubscribe python {fname} 3'

    try:
        subprocess.run(command.split(), check=True, stderr=True, env=os.environ)
    except subprocess.CalledProcessError:
        assert False
