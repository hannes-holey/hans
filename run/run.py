#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from eos.eos import DowsonHigginson
from solver.solver import Solver


class Run:

    def __init__(self, options, disc, geometry, numerics, material):

        self.material = material
        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics

        self.name = str(options['name'])

        self.plotDim = int(options['plotDim'])
        self.plotOption = int(options['plotOption'])
        self.plotInterval = int(options['plotInterval'])
        self.writeOutput = int(options['writeOutput'])
        self.writeInterval = int(options['writeInterval'])

        self.maxIt = int(numerics['maxIt'])

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        self.sol = Solver(options, disc, geometry, numerics, material)

        if self.plotOption == 0:
            for i in range(self.maxIt):
                self.sol.solve(i)
        else:
            self.plot()

    def plot(self):

        if self.plotDim == 1:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (14,9), sharex=True)

            x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

            self.line0, = self.ax1[0,0].plot(x, self.sol.q.field[0][:,int(self.Ny/2)])
            self.line1, = self.ax1[0,1].plot(x, self.sol.q.field[1][:,int(self.Ny/2)])
            self.line2, = self.ax1[1,0].plot(x, self.sol.q.field[2][:,int(self.Ny/2)])
            self.line3, = self.ax1[1,1].plot(x, DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            self.ax1[1,0].set_xlabel('distance x (m)')
            self.ax1[1,1].set_xlabel('distance x (m)')

            self.limits = np.zeros((4,3))

            for j in range(3):
                self.limits[j,0] = np.amin(self.sol.q.field[j][:,int(self.Ny/2)])
                self.limits[j,1] = np.amax(self.sol.q.field[j][:,int(self.Ny/2)])

            self.limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))
            self.limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2][:,int(self.Ny/2)]))

            ani = animation.FuncAnimation(self.fig, self.animate1D, self.maxIt, fargs=(self.sol,), interval=1 ,repeat=False)

        elif self.plotDim == 2:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (12,9), sharex=True, sharey=True, tight_layout=False)

            self.im0 = self.ax1[0,0].imshow(self.sol.q.field[0], interpolation='nearest')
            self.im1 = self.ax1[0,1].imshow(self.sol.q.field[1], interpolation='nearest')
            self.im2 = self.ax1[1,0].imshow(self.sol.q.field[2], interpolation='nearest')
            self.im3 = self.ax1[1,1].imshow(DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2]), interpolation='nearest')

            self.cbar0 = plt.colorbar(self.im0, ax = self.ax1[0,0])
            self.cbar1 = plt.colorbar(self.im1, ax = self.ax1[0,1])
            self.cbar2 = plt.colorbar(self.im2, ax = self.ax1[1,0])
            self.cbar3 = plt.colorbar(self.im3, ax = self.ax1[1,1])

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            self.limits = np.zeros((4,3))

            for j in range(3):
                self.limits[j,0] = np.amin(self.sol.q.field[j])
                self.limits[j,1] = np.amax(self.sol.q.field[j])

            self.limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2]))
            self.limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(self.sol.q.field[2]))

            ani = animation.FuncAnimation(self.fig, self.animate2D, self.maxIt, fargs=(self.sol,), interval=1, repeat=False)

        elif self.plotDim == 3:
            self.fig, self.ax1 = plt.subplots(1,1, figsize = (12,9), sharex=True, sharey=True, tight_layout=False)

            self.im0 = self.ax1.streamplot(self.sol.q.xx[:,0], self.sol.q.yy[0,:], self.sol.q.field[0], self.sol.q.field[1])
            self.ax1.set_title(r'$j$')

            ani = animation.FuncAnimation(self.fig, self.animate2D_stream, self.maxIt, fargs=(self.sol,), interval=1, repeat=False)

        if self.plotOption == 1:
            plt.show()
        elif self.plotOption == 2:
            ani.save('./output/animations/'+ self.name + '_' + str(self.sol.ani_tag).zfill(4) + '.mp4',fps=30)

    def animate1D(self, i, sol):

        if i%self.plotInterval == 0:

            for j in range(3):
                if np.amin(sol.q.field[j][:,int(self.Ny/2)]) < self.limits[j,0]:
                    self.limits[j,0] = np.amin(sol.q.field[j][:,int(self.Ny/2)])
                if np.amax(sol.q.field[j][:,int(self.Ny/2)]) > self.limits[j,1]:
                    self.limits[j,1] = np.amax(sol.q.field[j][:,int(self.Ny/2)])

            if np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)])) < self.limits[3,0]:
                self.limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))
            if np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)])) > self.limits[3,1]:
                self.limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))

            # buffer
            for j in range(4):
                if self.limits[j,1] == self.limits[j,0] and self.limits[j,0] != 0.:
                    self.limits[j,2] = 0.5*self.limits[j,1]
                elif self.limits[j,1] == self.limits[j,0] and self.limits[j,0] == 0.:
                    self.limits[j,2] = 1.
                else:
                    self.limits[j,2] = 0.1*(self.limits[j,1] - self.limits[j,0])

            self.fig.suptitle('step = %1d' % (i))

            self.ax1[0,0].set_ylim(self.limits[0,0] - self.limits[0,2] , self.limits[0,1] + self.limits[0,2])
            self.ax1[0,1].set_ylim(self.limits[1,0] - self.limits[1,2] , self.limits[1,1] + self.limits[1,2])
            self.ax1[1,0].set_ylim(self.limits[2,0] - self.limits[2,2] , self.limits[2,1] + self.limits[2,2])
            self.ax1[1,1].set_ylim(self.limits[3,0] - self.limits[3,2] , self.limits[3,1] + self.limits[3,2])

            self.line0.set_ydata(sol.q.field[0][:,int(self.Ny/2)])
            self.line1.set_ydata(sol.q.field[1][:,int(self.Ny/2)])
            self.line2.set_ydata(sol.q.field[2][:,int(self.Ny/2)])
            self.line3.set_ydata(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))

        sol.solve(i)

    def animate2D(self, i, sol):

        if i%self.plotInterval == 0:

            for j in range(3):
                if np.amin(sol.q.field[j]) < self.limits[j,0]:
                    self.limits[j,0] = np.amin(sol.q.field[j])
                if np.amax(sol.q.field[j]) > self.limits[j,1]:
                    self.limits[j,1] = np.amax(sol.q.field[j])

            if np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2])) < self.limits[3,0]:
                self.limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2]))
            if np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2])) > self.limits[3,1]:
                self.limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2]))

            # buffer
            for j in range(4):
                if self.limits[j,1] == self.limits[j,0]:
                    self.limits[j,2] = 0.5*self.limits[j,1]
                else:
                    self.limits[j,2] = 0.1*(self.limits[j,1] - self.limits[j,0])


            self.fig.suptitle('step = %1d' % (i))

            self.im0.set_clim(vmin = self.limits[0,0] - self.limits[0,2],vmax = self.limits[0,1] + self.limits[0,2])
            self.im1.set_clim(vmin = self.limits[1,0] - self.limits[1,2],vmax = self.limits[1,1] + self.limits[1,2])
            self.im2.set_clim(vmin = self.limits[2,0] - self.limits[2,2],vmax = self.limits[2,1] + self.limits[2,2])
            self.im3.set_clim(vmin = self.limits[3,0] - self.limits[3,2],vmax = self.limits[3,1] + self.limits[3,2])

            self.im0.set_array(sol.q.field[0].T)
            self.im1.set_array(sol.q.field[1].T)
            self.im2.set_array(sol.q.field[2].T)
            self.im3.set_array(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2].T))

        sol.solve(i)

    def animate2D_stream(self, i, sol):

        if i%self.plotInterval == 0:

            self.ax1.collections = [] # clear lines streamplot
            self.ax1.patches = [] # clear arrowheads streamplot

            self.fig.suptitle('step = %1d' % (i))

            stream = self.ax1.streamplot(self.sol.q.xx[:,0], self.sol.q.yy[0,:], self.sol.q.field[0], self.sol.q.field[1], color = 'blue')
            # self.fig.colorbar(stream.lines)

            self.ax1.set_xlim(0, self.Lx)
            self.ax1.set_ylim(0, self.Ly)

        sol.solve(i)

        return stream
