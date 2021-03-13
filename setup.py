from setuptools import setup, find_packages

setup(name='pylub',
      description='A 2D Fluid Mechanics Solver for Lubrication',
      url='http://github.com/hannes-holey/pylub',
      author='Hannes Holey',
      author_email='hannes.holey@kit.edu',
      license="MIT",
      packages=find_packages(),
      scripts=['cli/plot1D_evolution.py',
               'cli/plot1D_last.py',
               'cli/plot2D_last.py',
               'cli/plot_scalar.py',
               'cli/read_config.py',
               'cli/animate2d.py'],
      test_suite='tests',
      install_requires=["numpy>=1.18.1",
                        "matplotlib>=3.2.0",
                        "PyYAML>=5.3",
                        "GitPython>=3.1.0",
                        "netCDF4>=1.5.3"],
      zip_safe=False)
