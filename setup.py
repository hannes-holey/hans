"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


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
               'cli/animate1D.py',
               'cli/animate2D.py'],
      test_suite='tests',
      tests_require=["pytest>=4"],
      install_requires=requirements,
      python_requires=">=3.6",
      use_scm_version=True,
      setup_requires=['setuptools_scm>=3.5.0'],
      zip_safe=False)
