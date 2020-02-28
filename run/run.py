#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import h5py

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
        self.save_ani = bool(options['save_ani'])
        self.maxIt = int(options['maxIt'])
        self.plot_dim = int(options['plot_dim'])
        self.plotInterval = int(options['plotInterval'])
        self.writeOutput = int(options['writeOutput'])
        self.writeField = int(options['writeField'])

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        self.sol = Solver(options, disc, geometry, numerics, material)

        if self.writeOutput == True:
            self.tagO = 0
            while 'out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat' in os.listdir('./output'):
                self.tagO += 1
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%14s \t %14s \t %14s \t %14s \n" % ('time', 'mass', 'vmax', 'jXmax'))
        if self.writeField == True:
            self.tagF = 0
            while 'field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.h5' in os.listdir('./output'):
                self.tagF += 1

        self.plot()

    def plot(self):

        if self.plot_dim == 1:
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
            # self.ax1[1,1].plot(x, -1e-4*self.P_analytical(x)*self.dimless  + self.P0, '-')

        elif self.plot_dim == 2:
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

            ani = animation.FuncAnimation(self.fig, self.animate2D, self.maxIt, fargs=(self.sol,), interval=1, repeat=False)

        # elif self.plot_dim == 3:
        #     fig, self.ax1 = plt.subplots(1)
        #     # x = np.linspace(0, self.Lx, self.Nx, endpoint=True)
        #     z = np.linspace(0., 1.e-5, 50)
        #     # x = np.linspace(0, self.h1, 10, endpoint=True)
        #     # self.line, = self.ax1.plot((self.U/self.height.field[0][-1, int(self.Ny/2)] + field.field[comp][-1, int(self.Ny/2)]*z)*(self.height.field[0][-1, int(self.Ny/2)] - z),z)
        #     self.line, = self.ax1.plot(0*z,z)
        #     # self.line, = self.ax1.plot(x, field.field[comp])
        #     #self.ax1.plot(x, -self.P_analytical(x)*self.dimless  + self.P0, '-')
        #     self.time_text = self.ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
        #     ani = animation.FuncAnimation(fig, self.animate1D_z, self.maxIt, fargs=(field,), interval=1 ,repeat=False)
        #     self.ax1.grid()

        if self.save_ani == True:
            i = 0
            while str(self.name) + '_' + str(i).zfill(2) + '.mp4' in os.listdir('./output'):
                i += 1
            ani.save('./output/'+ self.name + '_' + str(i).zfill(2) + '.mp4',fps=30)
        else:
            plt.show()

    def animate1D(self, i, sol):

        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            # limits = np.empty((4,3))

            for j in range(3):
                if np.amin(sol.q.field[j][:,int(self.Ny/2)]) < self.limits[j,0]:
                    self.limits[j,0] = np.amin(sol.q.field[j][:,int(self.Ny/2)])
                if np.amax(sol.q.field[j][:,int(self.Ny/2)]) > self.limits[j,1]:
                    self.limits[j,1] = np.amax(sol.q.field[j][:,int(self.Ny/2)])

            if np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)])) < self.limits[3,0]:
                self.limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))
            if np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)])) > self.limits[3,1]:
                self.limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))

            # manual limits y-axis
            # limits[0,0] = 0.
            # limits[0,1] = 20.
            # limits[1,0] = 0.
            # limits[1,1] = 0.
            # limits[2,0] = 0.
            # limits[2,1] = 0.
            # limits[3,0] = 0.
            # limits[3,1] = 0.

            # buffer
            for j in range(4):
                if self.limits[j,1] == self.limits[j,0] and self.limits[j,0] != 0.:
                    self.limits[j,2] = 0.5*self.limits[j,1]
                elif self.limits[j,1] == self.limits[j,0] and self.limits[j,0] == 0.:
                    self.limits[j,2] = 1.
                else:
                    self.limits[j,2] = 0.1*(self.limits[j,1] - self.limits[j,0])

            self.ax1[0,0].set_ylim(self.limits[0,0] - self.limits[0,2] , self.limits[0,1] + self.limits[0,2])
            self.ax1[0,1].set_ylim(self.limits[1,0] - self.limits[1,2] , self.limits[1,1] + self.limits[1,2])
            self.ax1[1,0].set_ylim(self.limits[2,0] - self.limits[2,2] , self.limits[2,1] + self.limits[2,2])
            self.ax1[1,1].set_ylim(self.limits[3,0] - self.limits[3,2] , self.limits[3,1] + self.limits[3,2])

            self.line0.set_ydata(sol.q.field[0][:,int(self.Ny/2)])
            self.line1.set_ydata(sol.q.field[1][:,int(self.Ny/2)])
            self.line2.set_ydata(sol.q.field[2][:,int(self.Ny/2)])
            self.line3.set_ydata(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))

            self.fig.suptitle('step = %1d' % (i))

            if self.writeField == True:

                # HDF5 output file

                out0 = sol.q.field[0]
                out1 = sol.q.field[1]
                out2 = sol.q.field[2]
                out3 = DowsonHigginson(self.material).isoT_pressure(sol.q.field[2])

                file = h5py.File('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.h5', 'a')

                if '/step'+ str(i).zfill(len(str(self.maxIt))) not in file:

                    g1 =file.create_group('step'+ str(i).zfill(len(str(self.maxIt))))

                    g1.create_dataset('j_x',   data=out0)
                    g1.create_dataset('j_y',   data=out1)
                    g1.create_dataset('rho',   data=out2)
                    g1.create_dataset('press', data=out3)

                    g1.attrs.create('time', sol.dt*i)
                    g1.attrs.create('mass', sol.mass)

                file.close()

        sol.solve(i)

    def animate2D(self, i, sol):

        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            limits = np.empty((4,3))

            for j in range(3):
                limits[j,0] = np.amin(sol.q.field[j][:,int(self.Ny/2)])
                limits[j,1] = np.amax(sol.q.field[j][:,int(self.Ny/2)])

            limits[3,0] = np.amin(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2][:,int(self.Ny/2)]))

            # manual limits y-axis
            # limits[0,0] = 40.
            # limits[0,1] = 45.
            # limits[1,0] = 0.
            # limits[1,1] = 0.
            # limits[2,0] = 0.
            # limits[2,1] = 0.
            # limits[3,0] = 0.
            # limits[3,1] = 0.

            # buffer
            for j in range(4):
                if limits[j,1] == limits[j,0]:
                    limits[j,2] = 0.5*limits[j,1]
                else:
                    limits[j,2] = 0.1*(limits[j,1] - limits[j,0])


            self.fig.suptitle('step = %1d' % (i))

            self.im0.set_clim(vmin = limits[0,0] - limits[0,2],vmax = limits[0,1] + limits[0,2])
            self.im1.set_clim(vmin = limits[1,0] - limits[1,2],vmax = limits[1,1] + limits[1,2])
            self.im2.set_clim(vmin = limits[2,0] - limits[2,2],vmax = limits[2,1] + limits[2,2])
            self.im3.set_clim(vmin = limits[3,0] - limits[3,2],vmax = limits[3,1] + limits[3,2])

            self.im0.set_array(sol.q.field[0].T)
            self.im1.set_array(sol.q.field[1].T)
            self.im2.set_array(sol.q.field[2].T)
            self.im3.set_array(DowsonHigginson(self.material).isoT_pressure(sol.q.field[2].T))


            # lower = np.amin(field.field[comp])
            # upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            # lower = -50.
            # upper = 0.0

            # v1 = np.linspace(lower, upper, 11, endpoint=True)
            # self.cbar.set_ticks(v1)
            # self.cbar.draw_all()
            # self.time_text.set_text('time = %.6f ms' % (i*self.dt *1e3))

        sol.solve(i)

    def animate1D_z(self, i, field, comp):
        if i%self.plotInterval == 0:
            # adaptive bounds
            # lower = np.amin(field.field[comp][0, int(self.Ny/2)]*self.h1**2/4.)
            # upper = np.amax(field.field[comp][:, int(self.Ny/2)]*self.h1**2/4.)

            # inclined
            lower = -0.5
            upper = 2.

            # poiseuille
            #lower = -0.000
            #upper = 2e-3

            if upper == lower:
                vspace = 0.5*upper
            else:
                vspace = 0.1*(upper - lower)
            self.time_text.set_text('step = %.1f' % (i) )
            # self.time_text.set_text('time = %.3f µs' % (i*self.dt * 1e6) )

            # right
            z = np.linspace(0., self.height.field[0][-1, int(self.Ny/2)], 50)
            self.line.set_ydata(z)
            self.line.set_xdata(self.U/self.height.field[0][-1, int(self.Ny/2)]*z + field.field[comp][-1, int(self.Ny/2)]*z*(self.height.field[0][-1, int(self.Ny/2)] - z))
            # left
            # z = np.linspace(0., self.height.field[0][0, int(self.Ny/2)], 50)
            # self.line.set_xdata(self.U/self.height.field[0][0, int(self.Ny/2)]*z + field.field[comp][0, int(self.Ny/2)]*z*(self.height.field[0][0, int(self.Ny/2)] - z))
            # middle
            # z = np.linspace(0., self.height.field[0][int(self.Nx/2), int(self.Ny/2)], 50)
            # self.line.set_xdata(self.U/self.height.field[0][int(self.Nx/2), int(self.Ny/2)]*z + field.field[comp][int(self.Nx/2), int(self.Ny/2)]*z*(self.height.field[0][int(self.Nx/2), int(self.Ny/2)] - z))


            self.ax1.set_xlim(lower - vspace , upper + vspace)
            self.ax1.set_ylim(0., self.h2)

            if self.writeField == True:
                out = np.vstack((field.xx[:,int(self.Ny/2)],field.field[comp][:,int(self.velNy/2)])).T
                with open('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat', "a+") as f:
                    np.savetxt(f, out, header='step' + str(i))
                #     f.write(str(i) + '\n' + str(field.field[comp]) + '\n')
        self.solve(i)
