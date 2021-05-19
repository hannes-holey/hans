from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setup(name='hans',
      description='Height-Averaged Navier-Stokes (HANS) solver for 2D lubrication problems',
      author='Hannes Holey',
      author_email='hannes.holey@kit.edu',
      url='http://github.com/hannes-holey/hans',
      license="MIT",
      packages=find_packages(),
      package_data={'': ['ChangeLog.md']},
      include_package_data=True,
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
                        "netCDF4>=1.5.3",
                        "pytest>4"],
      zip_safe=False)
