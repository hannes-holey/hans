#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from eos.eos import DowsonHigginson, PowerLaw
from solver.solver import Solver

class Run:

    def __init__(self, options, disc, geometry, numerics, material):

        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

        plotOption = bool(options['plot'])
        self.writeInterval = int(options['writeInterval'])
        tol = float(numerics['tol'])
        self.name = str(geometry['name'])

        self.maxIt= int(numerics['maxT'] * 1e9 /numerics['dt'])

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        if material['EOS'] == 'DH':
            self.eqOfState = DowsonHigginson(material)
        elif material['EOS'] == 'PL':
            self.eqOfState = PowerLaw(material)

        self.sol = Solver(disc, geometry, numerics, material)

        if plotOption == False:
            self.file_tag = 1

            if 'output' not in os.listdir():
                os.mkdir('output')

            while str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5' in os.listdir('output'):
                self.file_tag += 1

            outfile = str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5'

            i = 0

            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Timestep", "Time", "Delta_v"))
            while self.sol.time * 1e9 < numerics['maxT']:

                self.sol.solve(i)
                self.write(i, 0)
                if i % self.writeInterval == 0:
                    print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))
                #print("Simulation time : {:.2f} ns / {:<d} ns".format(self.sol.time * 1e9, numerics['maxT']), end = "\r")
                i += 1

                tDiff = numerics['maxT'] * 1e-9 - self.sol.time

                if self.sol.eps < tol and self.sol.time > 1e-6:
                    self.write(i, 1)
                    #print("Simulation time : {:.2f} ns / {:<d} ns".format(self.sol.time * 1e9, numerics['maxT']), end = "\n")
                    print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))
                    print("\nSolution has converged after {:d} steps, Output written to : {:s}".format(i, outfile))
                    break
                elif tDiff < self.sol.dt:
                    self.sol.solve(i)
                    self.write(i, 1)
                    # print("Simulation time : {:.2f} ns / {:<d} ns".format(self.sol.time * 1e9, numerics['maxT']), end = "\n")
                    print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))
                    print("\nNo convergence within {:d} steps. Stopping criterion: maximum time {:d} ns reached.".format(i, numerics['maxT']))
                    print("Output written to : {:s}".format(outfile))

        else:
            self.plot()

    def write(self, i, last):

        # HDF5 output file
        if i % self.writeInterval == 0 or last == 1:

            file = h5py.File('./output/' + str(self.name) + '_' + str(self.file_tag).zfill(4) + '.h5', 'a')

            if 'config' not in file:
                g0 = file.create_group('config')

                from datetime import datetime

                now = datetime.now()
                timeString = now.strftime("%d/%m/%Y %H:%M:%S")

                g0.attrs.create("Start time:",  timeString)

                categories = {'options': self.options,
                              'disc': self.disc,
                              'geometry': self.geometry,
                              'numerics': self.numerics,
                              'material': self.material}

                for cat_key, cat_val in categories.items():
                    g1 = file.create_group('config/' + cat_key)

                    for key, value in cat_val.items():
                        g1.attrs.create(str(key), value)

            if str(i).zfill(10) not in file:

                g1 =file.create_group(str(i).zfill(10))

                g1.create_dataset('j_x',   data = self.sol.q.field[0])
                g1.create_dataset('j_y',   data = self.sol.q.field[1])
                g1.create_dataset('rho',   data = self.sol.q.field[2])
                g1.create_dataset('press', data = self.eqOfState.isoT_pressure(self.sol.q.field[2]))

                g1.attrs.create('time', self.sol.time)
                g1.attrs.create('mass', self.sol.mass)
                g1.attrs.create('vmax', self.sol.vmax)
                g1.attrs.create('vSound', self.sol.vSound)
                g1.attrs.create('dt', self.sol.dt)
                g1.attrs.create('eps', self.sol.eps)

            file.close()

    def plot(self):

        fig, ax = plt.subplots(2,2, figsize = (14,9), sharex=True)
        x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

        line0, = ax[0,0].plot(x, self.sol.q.field[0][:,int(self.Ny/2)])
        line1, = ax[0,1].plot(x, self.sol.q.field[1][:,int(self.Ny/2)])
        line2, = ax[1,0].plot(x, self.sol.q.field[2][:,int(self.Ny/2)])
        line3, = ax[1,1].plot(x, self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

        ax[0,0].set_title(r'$j_x$')
        ax[0,1].set_title(r'$j_y$')
        ax[1,0].set_title(r'$\rho$')
        ax[1,1].set_title(r'$p$')

        ax[1,0].set_xlabel('distance x (m)')
        ax[1,1].set_xlabel('distance x (m)')

        limits = np.zeros((4,3))

        for j in range(3):
            limits[j,0] = np.amin(self.sol.q.field[j][:,int(self.Ny/2)])
            limits[j,1] = np.amax(self.sol.q.field[j][:,int(self.Ny/2)])

        limits[3,0] = np.amin(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))
        limits[3,1] = np.amax(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

        ani = animation.FuncAnimation(fig, self.animate1D, 100000,
                                    fargs=(fig, ax, line0, line1, line2, line3, limits,), interval=1 ,repeat=False)

        plt.show()

    def animate1D(self, i, fig, ax, line0, line1, line2, line3, limits):

        limits = self.adaptiveLimits(limits)

        fig.suptitle('time = {:.2f} ns'.format(self.sol.time * 1e9))

        ax[0,0].set_ylim(limits[0,0] - limits[0,2] , limits[0,1] + limits[0,2])
        ax[0,1].set_ylim(limits[1,0] - limits[1,2] , limits[1,1] + limits[1,2])
        ax[1,0].set_ylim(limits[2,0] - limits[2,2] , limits[2,1] + limits[2,2])
        ax[1,1].set_ylim(limits[3,0] - limits[3,2] , limits[3,1] + limits[3,2])

        line0.set_ydata(self.sol.q.field[0][:,int(self.Ny/2)])
        line1.set_ydata(self.sol.q.field[1][:,int(self.Ny/2)])
        line2.set_ydata(self.sol.q.field[2][:,int(self.Ny/2)])
        line3.set_ydata(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

        self.sol.solve(i)

    def adaptiveLimits(self, limits):

        for j in range(3):
            if np.amin(self.sol.q.field[j][:,int(self.Ny/2)]) < limits[j,0]:
                limits[j,0] = np.amin(self.sol.q.field[j][:,int(self.Ny/2)])
            if np.amax(self.sol.q.field[j][:,int(self.Ny/2)]) > limits[j,1]:
                limits[j,1] = np.amax(self.sol.q.field[j][:,int(self.Ny/2)])

        if np.amin(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)])) < limits[3,0]:
            limits[3,0] = np.amin(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))
        if np.amax(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)])) > limits[3,1]:
            limits[3,1] = np.amax(self.eqOfState.isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

        for j in range(4):
            if limits[j,1] == limits[j,0] and limits[j,0] != 0.:
                limits[j,2] = 0.5*limits[j,1]
            elif limits[j,1] == limits[j,0] and limits[j,0] == 0.:
                limits[j,2] = 1.
            else:
                limits[j,2] = 0.1*(limits[j,1] - limits[j,0])

        return limits
