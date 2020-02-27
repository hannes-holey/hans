#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import h5py

from eos.eos import DowsonHigginson
from geo.geometry import Analytic
from field.field import scalarField
from field.field import vectorField
from field.field import tensorField

class LaxFriedrichs:

    def __init__(self, options, geometry, numerics, material, BC):

        self.name = str(options['name'])
        self.save_ani = bool(options['save_ani'])
        self.maxIt = int(options['maxIt'])
        self.plot_dim = int(options['plot_dim'])
        self.plotInterval = int(options['plotInterval'])
        self.writeOutput = int(options['writeOutput'])
        self.writeField = int(options['writeField'])

        self.Lx = float(geometry['Lx'])
        self.Ly = float(geometry['Ly'])
        self.h1 = float(geometry['h1'])
        self.h2 = float(geometry['h2'])

        self.Nx = int(numerics['Nx'])
        self.Ny = int(numerics['Ny'])
        self.dt = float(numerics['dt'])
        self.freq = int(numerics['freq'])

        self.mu = float(material['mu'])
        self.s = float(material['s'])

        # Stokes assumption
        self.lam = -2./3. * self.mu

        self.U = float(BC['U'])
        self.V = float(BC['V'])
        self.P0 = float(BC['P0'])
        self.rho0 = float(BC['rho0'])
        self.temp = float(BC['temp'])

        # Gap height
        self.height = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        if self.name == 'journal':
            self.height.fromFunctionXY(Analytic(geometry).journalBearing, 0)
        elif self.name == 'inclined' or self.name == 'poiseuille':
            self.height.fromFunctionXY(Analytic(geometry).linearSlider, 0)

        self.height.getGradients()

        self.q = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.q.field[2] = self.rho0 * np.ones(shape=(self.Nx, self.Ny))

        if self.name == 'inclined':
            self.q.field[2][0,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
            self.q.field[2][-1,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
        elif self.name == 'poiseuille':
            self.q.field[2][-1,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
            self.q.field[2][0,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(2. * self.P0)

        self.fluxX_E = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxX_W = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxY_N = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxY_S = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.rhs = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Stress
        self.stress = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.vel = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.alp = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

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

        self.plot(self.q)

    def solve(self, i):

        if i%self.freq == 0:
            if self.name == 'journal':
                self.stress.getStressNewtonian_Rey(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
                # self.stress.getStressNewtonian_avg4(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
            elif self.name == 'inclined':
                self.stress.getStressNewtonian_avg4(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
            elif self.name == 'poiseuille':
                self.stress.getStressNewtonian_Rey(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)

            #self.stress.addNoise(self.s)

        if self.name == 'journal':
            periodic = 1
        else:
            periodic = 0

        # LF fluxes
        self.fluxX_E.fluxX_LF(self.stress, self.q, self.dt, -1, periodic)
        self.fluxX_W.fluxX_LF(self.stress, self.q, self.dt,  1, periodic)

        self.fluxY_N.fluxY_LF(self.stress, self.q, self.dt, -1, periodic)
        self.fluxY_S.fluxY_LF(self.stress, self.q, self.dt,  1, periodic)


        # Dirichlet BCs (rho)
        if periodic == 0:
            self.fluxX_W.field[2][0,:]  = self.fluxX_E.field[2][0,:]
            self.fluxX_E.field[2][-1,:] = self.fluxX_W.field[2][-1,:]
            self.fluxY_S.field[2][0,:]  = self.fluxY_N.field[2][0,:]
            self.fluxY_N.field[2][-1,:] = self.fluxY_S.field[2][-1,:]

        # Periodic BCs (flux)
        # self.fluxX_W.field[0][0,:] = self.fluxX_E.field[0][-1,:]
        # self.fluxX_W.field[1][0,:] = self.fluxX_E.field[1][-1,:]
        # self.fluxX_W.field[2][0,:] = self.fluxX_E.field[2][-1,:]

        # self.fluxY_N.field[0][:,-1] = self.fluxY_S.field[0][:,0]
        # self.fluxY_N.field[1][:,-1] = self.fluxY_S.field[1][:,0]
        # self.fluxY_N.field[2][:,-1] = self.fluxY_S.field[2][:,0]

        # RHS + wall stress correction
        self.rhs.computeRHS(self.fluxX_E, self.fluxX_W, self.fluxY_N, self.fluxY_S)
        self.rhs.addStress_wall(self.height, self.q, self.mu, self.U, self.V)

        # print(self.fluxX_E.field[2][int(self.Nx/4),int(self.Ny/2)], self.fluxX_W.field[2][int(self.Nx/4),int(self.Ny/2)], self.q.field[2][int(self.Nx/4),int(self.Ny/2)], self.rhs.field[2][int(self.Nx/4),int(self.Ny/2)])

        # explicit time step
        self.q.update_explicit(self.rhs, self.dt)


        self.vel.field[0] = self.q.field[1]/self.q.field[2]
        self.alp.field[0] =  6.*self.mu/(self.q.field[2] * self.height.field[0]**2) * (self.U*self.q.field[2] - 2.* self.q.field[0])

        # self.mass = np.sum(self.q.field[2] * self.height.field[0])*self.Lx*self.Ly
        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)

        if self.writeOutput == True:
            mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
            vmax = np.amax(np.abs(self.vel.field[0]))
            maxFlux_X = np.amax(self.q.field[0])
            netFluxX = np.sum(self.q.field[0])
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%.8e \t %.8e \t %.8e \t %.8e \n" % (i*self.dt, mass, vmax, netFluxX))

    def plot(self, field):

        if self.plot_dim == 1:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (14,9), sharex=True)

            x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

            self.line0, = self.ax1[0,0].plot(x, field.field[0][:,int(self.Ny/2)])
            self.line1, = self.ax1[0,1].plot(x, field.field[1][:,int(self.Ny/2)])
            self.line2, = self.ax1[1,0].plot(x, field.field[2][:,int(self.Ny/2)])
            self.line3, = self.ax1[1,1].plot(x, DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            ani = animation.FuncAnimation(self.fig, self.animate1D, self.maxIt, fargs=(field,), interval=1 ,repeat=False)

            self.ax1[1,0].set_xlabel('distance x (m)')
            self.ax1[1,1].set_xlabel('distance x (m)')

            # self.ax1[1,1].plot(x, -1e-4*self.P_analytical(x)*self.dimless  + self.P0, '-')

        elif self.plot_dim == 2:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (12,9), sharex=True, sharey=True, tight_layout=False)

            self.im0 = self.ax1[0,0].imshow(field.field[0], interpolation='nearest')
            self.im1 = self.ax1[0,1].imshow(field.field[1], interpolation='nearest')
            self.im2 = self.ax1[1,0].imshow(field.field[2], interpolation='nearest')
            self.im3 = self.ax1[1,1].imshow(field.field[2], interpolation='nearest')

            self.cbar0 = plt.colorbar(self.im0, ax = self.ax1[0,0])
            self.cbar1 = plt.colorbar(self.im1, ax = self.ax1[0,1])
            self.cbar2 = plt.colorbar(self.im2, ax = self.ax1[1,0])
            self.cbar3 = plt.colorbar(self.im3, ax = self.ax1[1,1])

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            ani = animation.FuncAnimation(self.fig, self.animate2D, self.maxIt, fargs=(field,), interval=1, repeat=False)

        elif self.plot_dim == 3:
            fig, self.ax1 = plt.subplots(1)
            # x = np.linspace(0, self.Lx, self.Nx, endpoint=True)
            z = np.linspace(0., 1.e-5, 50)
            # x = np.linspace(0, self.h1, 10, endpoint=True)
            # self.line, = self.ax1.plot((self.U/self.height.field[0][-1, int(self.Ny/2)] + field.field[comp][-1, int(self.Ny/2)]*z)*(self.height.field[0][-1, int(self.Ny/2)] - z),z)
            self.line, = self.ax1.plot(0*z,z)
            # self.line, = self.ax1.plot(x, field.field[comp])
            #self.ax1.plot(x, -self.P_analytical(x)*self.dimless  + self.P0, '-')
            self.time_text = self.ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
            ani = animation.FuncAnimation(fig, self.animate1D_z, self.maxIt, fargs=(field,), interval=1 ,repeat=False)
            self.ax1.grid()

        if self.save_ani == True:
            i = 0
            while str(self.name) + '_' + str(i).zfill(2) + '.mp4' in os.listdir('./output'):
                i += 1
            ani.save('./output/'+ self.name + '_' + str(i).zfill(2) + '.mp4',fps=30)
        else:
            plt.show()

    def animate1D(self, i, field):

        self.solve(i)

        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            limits = np.empty((4,3))

            for j in range(3):
                limits[j,0] = np.amin(field.field[j][:,int(self.Ny/2)])
                limits[j,1] = np.amax(field.field[j][:,int(self.Ny/2)])

            limits[3,0] = np.amin(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

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
                if limits[j,1] == limits[j,0] and limits[j,0] != 0.:
                    limits[j,2] = 0.5*limits[j,1]
                elif limits[j,1] == limits[j,0] and limits[j,0] == 0.:
                    limits[j,2] = 1.
                else:
                    limits[j,2] = 0.1*(limits[j,1] - limits[j,0])

            self.ax1[0,0].set_ylim(limits[0,0] - limits[0,2] , limits[0,1] + limits[0,2])
            self.ax1[0,1].set_ylim(limits[1,0] - limits[1,2] , limits[1,1] + limits[1,2])
            self.ax1[1,0].set_ylim(limits[2,0] - limits[2,2] , limits[2,1] + limits[2,2])
            self.ax1[1,1].set_ylim(limits[3,0] - limits[3,2] , limits[3,1] + limits[3,2])

            self.line0.set_ydata(field.field[0][:,int(self.Ny/2)])
            self.line1.set_ydata(field.field[1][:,int(self.Ny/2)])
            self.line2.set_ydata(field.field[2][:,int(self.Ny/2)])
            self.line3.set_ydata(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

            self.fig.suptitle('step = %1d' % (i))

            if self.writeField == True:

                # HDF5 output file

                out0 = field.field[0]
                out1 = field.field[1]
                out2 = field.field[2]
                out3 = self.DowsHigg_isoT_P(field.field[2])

                file = h5py.File('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.h5', 'a')

                if '/step'+ str(i).zfill(len(str(self.maxIt))) not in file:

                    g1 =file.create_group('step'+ str(i).zfill(len(str(self.maxIt))))

                    g1.create_dataset('j_x', data=out0)
                    g1.create_dataset('j_y', data=out1)
                    g1.create_dataset('rho', data=out2)
                    g1.create_dataset('press', data=out3)

                    g1.attrs.create('time', self.dt*i)
                    g1.attrs.create('mass', self.mass)

                file.close()

    def animate2D(self, i, field):
        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            limits = np.empty((4,3))

            for j in range(3):
                limits[j,0] = np.amin(field.field[j][:,int(self.Ny/2)])
                limits[j,1] = np.amax(field.field[j][:,int(self.Ny/2)])

            limits[3,0] = np.amin(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

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

            self.im0.set_array(field.field[0].T)
            self.im1.set_array(field.field[1].T)
            self.im2.set_array(field.field[2].T)
            self.im3.set_array(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2].T))


            # lower = np.amin(field.field[comp])
            # upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            # lower = -50.
            # upper = 0.0

            # v1 = np.linspace(lower, upper, 11, endpoint=True)
            # self.cbar.set_ticks(v1)
            # self.cbar.draw_all()
            # self.time_text.set_text('time = %.6f ms' % (i*self.dt *1e3))

        self.solve(i)

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

class LaxWendroff:

    def __init__(self, options, geometry, numerics, material, BC):

        self.name = str(options['name'])
        self.save_ani = bool(options['save_ani'])
        self.maxIt = int(options['maxIt'])
        self.plot_dim = int(options['plot_dim'])
        self.plotInterval = int(options['plotInterval'])
        self.writeOutput = int(options['writeOutput'])
        self.writeField = int(options['writeField'])

        self.Lx = float(geometry['Lx'])
        self.Ly = float(geometry['Ly'])
        self.h1 = float(geometry['h1'])
        self.h2 = float(geometry['h2'])

        self.Nx = int(numerics['Nx'])
        self.Ny = int(numerics['Ny'])
        self.dt = float(numerics['dt'])
        self.freq = int(numerics['freq'])
        self.periodicX = bool(numerics['periodicX'])
        self.periodicY = bool(numerics['periodicY'])

        self.mu = float(material['mu'])
        self.s = float(material['s'])

        # Stokes assumption
        self.lam = -2./3. * self.mu

        self.U = float(BC['U'])
        self.V = float(BC['V'])
        self.P0 = float(BC['P0'])
        self.rho0 = float(BC['rho0'])
        self.temp = float(BC['temp'])

        # Gap height
        self.height = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.height_edges = vectorField(self.Nx + 1, self.Ny + 1, self.Lx, self.Ly)
        self.height_edges.edgesField()

        if self.name == 'journal':
            self.height.fromFunctionXY(Analytic(geometry).journalBearing, 0)
            self.height_edges.fromFunctionXY(Analytic(geometry).journalBearing,0)
        elif self.name == 'inclined' or self.name == 'poiseuille':
            self.height.fromFunctionXY(Analytic(geometry).linearSlider, 0)
            self.height_edges.fromFunctionXY(Analytic(geometry).linearSlider, 0)

        self.height.getGradients()
        self.height_edges.getGradients()

        # Conservative variables q + initial density
        self.q = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.q.field[2] = self.rho0 * np.ones(shape=(self.Nx, self.Ny))

        if self.name == 'inclined':
            self.q.field[2][0,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
            self.q.field[2][-1,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
        elif self.name == 'poiseuille':
            self.q.field[2][-1,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(self.P0)
            self.q.field[2][0,:] = DowsonHigginson(self.rho0, self.P0).isoT_density(2. * self.P0)

        # Conserved variables at cell edges
        self.Q_E = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.Q_W = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.Q_N = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.Q_S = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Stress at cell center
        self.stress = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # stress at cell edges
        self.stress_E = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.stress_W = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.stress_N = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.stress_S = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.rhs = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.vel = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.alp = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

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

        self.plot(self.q)

    def solve(self, i):

    # Richtmyer two-step method
    # 1st (Lax) step

        # Compute q at cell edges

        self.Q_E.QX_LF(self.stress, self.q, self.dt, -1, self.periodicX)
        self.Q_W.QX_LF(self.stress, self.q, self.dt,  1, self.periodicX)
        self.Q_N.QY_LF(self.stress, self.q, self.dt, -1, self.periodicY)
        self.Q_S.QY_LF(self.stress, self.q, self.dt,  1, self.periodicY)

        if i%self.freq == 0:
            if self.name == 'journal':
                self.stress.getStressNewtonian_Rey(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
                # self.stress.getStressNewtonian_avg4(self.height.field[0], self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
            elif self.name == 'inclined':
                self.stress.getStressNewtonian_avg4(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
            elif self.name == 'poiseuille':
                self.stress.getStressNewtonian_Rey(self.height, self.q, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)

            #self.stress.addNoise(self.s)

        # LF fluxes
        self.stress_E.getStressNewtonian_Rey(self.height, self.Q_E, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
        self.stress_W.getStressNewtonian_Rey(self.height, self.Q_W, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
        self.stress_N.getStressNewtonian_Rey(self.height, self.Q_N, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)
        self.stress_S.getStressNewtonian_Rey(self.height, self.Q_S, self.mu, self.lam, self.U, self.V, self.rho0, self.P0)

        # Dirichlet BCs (rho)
        # if self.periodicX == 0:
        #     self.fluxX_W.field[2][0,:]  = self.fluxX_E.field[2][0,:]
        #     self.fluxX_E.field[2][-1,:] = self.fluxX_W.field[2][-1,:]
        # elif self.periodicY == 0:
        #     self.fluxY_S.field[2][0,:]  = self.fluxY_N.field[2][0,:]
        #     self.fluxY_N.field[2][-1,:] = self.fluxY_S.field[2][-1,:]

        # Periodic BCs (flux)
        # self.fluxX_W.field[0][0,:] = self.fluxX_E.field[0][-1,:]
        # self.fluxX_W.field[1][0,:] = self.fluxX_E.field[1][-1,:]
        # self.fluxX_W.field[2][0,:] = self.fluxX_E.field[2][-1,:]

        # self.fluxY_N.field[0][:,-1] = self.fluxY_S.field[0][:,0]
        # self.fluxY_N.field[1][:,-1] = self.fluxY_S.field[1][:,0]
        # self.fluxY_N.field[2][:,-1] = self.fluxY_S.field[2][:,0]

        # RHS + wall stress correction
        self.rhs.computeRHS_LW(self.stress_E, self.stress_W, self.stress_N, self.stress_S, self.Q_E, self.Q_W, self.Q_N, self.Q_S)
        self.rhs.addStress_wall(self.height, self.q, self.mu, self.U, self.V)

        # print(self.fluxX_E.field[2][int(self.Nx/4),int(self.Ny/2)], self.fluxX_W.field[2][int(self.Nx/4),int(self.Ny/2)], self.q.field[2][int(self.Nx/4),int(self.Ny/2)], self.rhs.field[2][int(self.Nx/4),int(self.Ny/2)])

        # explicit time step
        self.q.update_explicit(self.rhs, self.dt)

        self.vel.field[0] = self.q.field[1]/self.q.field[2]
        self.alp.field[0] =  6.*self.mu/(self.q.field[2] * self.height.field[0]**2) * (self.U*self.q.field[2] - 2.* self.q.field[0])

        # self.mass = np.sum(self.q.field[2] * self.height.field[0])*self.Lx*self.Ly
        self.mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)

        if self.writeOutput == True:
            mass = np.sum(self.q.field[2] * self.height.field[0] * self.q.dx * self.q.dy)
            vmax = np.amax(np.abs(self.vel.field[0]))
            maxFlux_X = np.amax(self.q.field[0])
            netFluxX = np.sum(self.q.field[0])
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%.8e \t %.8e \t %.8e \t %.8e \n" % (i*self.dt, mass, vmax, netFluxX))

    def plot(self, field):

        if self.plot_dim == 1:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (14,9), sharex=True)

            x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

            self.line0, = self.ax1[0,0].plot(x, field.field[0][:,int(self.Ny/2)])
            self.line1, = self.ax1[0,1].plot(x, field.field[1][:,int(self.Ny/2)])
            self.line2, = self.ax1[1,0].plot(x, field.field[2][:,int(self.Ny/2)])
            self.line3, = self.ax1[1,1].plot(x, DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            ani = animation.FuncAnimation(self.fig, self.animate1D, self.maxIt, fargs=(field,), interval=1 ,repeat=False)

            self.ax1[1,0].set_xlabel('distance x (m)')
            self.ax1[1,1].set_xlabel('distance x (m)')

            # self.ax1[1,1].plot(x, -1e-4*self.P_analytical(x)*self.dimless  + self.P0, '-')

        elif self.plot_dim == 2:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (12,9), sharex=True, sharey=True, tight_layout=False)

            self.im0 = self.ax1[0,0].imshow(field.field[0], interpolation='nearest')
            self.im1 = self.ax1[0,1].imshow(field.field[1], interpolation='nearest')
            self.im2 = self.ax1[1,0].imshow(field.field[2], interpolation='nearest')
            self.im3 = self.ax1[1,1].imshow(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2]), interpolation='nearest')

            self.cbar0 = plt.colorbar(self.im0, ax = self.ax1[0,0])
            self.cbar1 = plt.colorbar(self.im1, ax = self.ax1[0,1])
            self.cbar2 = plt.colorbar(self.im2, ax = self.ax1[1,0])
            self.cbar3 = plt.colorbar(self.im3, ax = self.ax1[1,1])

            self.ax1[0,0].set_title(r'$j_x$')
            self.ax1[0,1].set_title(r'$j_y$')
            self.ax1[1,0].set_title(r'$\rho$')
            self.ax1[1,1].set_title(r'$p$')

            ani = animation.FuncAnimation(self.fig, self.animate2D, self.maxIt, fargs=(field,), interval=1, repeat=False)

        elif self.plot_dim == 3:
            fig, self.ax1 = plt.subplots(1)
            # x = np.linspace(0, self.Lx, self.Nx, endpoint=True)
            z = np.linspace(0., 1.e-5, 50)
            # x = np.linspace(0, self.h1, 10, endpoint=True)
            # self.line, = self.ax1.plot((self.U/self.height.field[0][-1, int(self.Ny/2)] + field.field[comp][-1, int(self.Ny/2)]*z)*(self.height.field[0][-1, int(self.Ny/2)] - z),z)
            self.line, = self.ax1.plot(0*z,z)
            # self.line, = self.ax1.plot(x, field.field[comp])
            #self.ax1.plot(x, -self.P_analytical(x)*self.dimless  + self.P0, '-')
            self.time_text = self.ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
            ani = animation.FuncAnimation(fig, self.animate1D_z, self.maxIt, fargs=(field,), interval=1 ,repeat=False)
            self.ax1.grid()

        if self.save_ani == True:
            i = 0
            while str(self.name) + '_' + str(i).zfill(2) + '.mp4' in os.listdir('./output'):
                i += 1
            ani.save('./output/'+ self.name + '_' + str(i).zfill(2) + '.mp4',fps=30)
        else:
            plt.show()

    def animate1D(self, i, field):

        self.solve(i)

        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            limits = np.empty((4,3))

            for j in range(3):
                limits[j,0] = np.amin(field.field[j][:,int(self.Ny/2)])
                limits[j,1] = np.amax(field.field[j][:,int(self.Ny/2)])

            limits[3,0] = np.amin(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

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
                if limits[j,1] == limits[j,0] and limits[j,0] != 0.:
                    limits[j,2] = 0.5*limits[j,1]
                elif limits[j,1] == limits[j,0] and limits[j,0] == 0.:
                    limits[j,2] = 1.
                else:
                    limits[j,2] = 0.1*(limits[j,1] - limits[j,0])

            self.ax1[0,0].set_ylim(limits[0,0] - limits[0,2] , limits[0,1] + limits[0,2])
            self.ax1[0,1].set_ylim(limits[1,0] - limits[1,2] , limits[1,1] + limits[1,2])
            self.ax1[1,0].set_ylim(limits[2,0] - limits[2,2] , limits[2,1] + limits[2,2])
            self.ax1[1,1].set_ylim(limits[3,0] - limits[3,2] , limits[3,1] + limits[3,2])

            self.line0.set_ydata(field.field[0][:,int(self.Ny/2)])
            self.line1.set_ydata(field.field[1][:,int(self.Ny/2)])
            self.line2.set_ydata(field.field[2][:,int(self.Ny/2)])
            self.line3.set_ydata(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

            self.fig.suptitle('step = %1d' % (i))

            if self.writeField == True:

                # HDF5 output file

                out0 = field.field[0]
                out1 = field.field[1]
                out2 = field.field[2]
                out3 = self.DowsHigg_isoT_P(field.field[2])

                file = h5py.File('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.h5', 'a')

                if '/step'+ str(i).zfill(len(str(self.maxIt))) not in file:

                    g1 =file.create_group('step'+ str(i).zfill(len(str(self.maxIt))))

                    g1.create_dataset('j_x', data=out0)
                    g1.create_dataset('j_y', data=out1)
                    g1.create_dataset('rho', data=out2)
                    g1.create_dataset('press', data=out3)

                    g1.attrs.create('time', self.dt*i)
                    g1.attrs.create('mass', self.mass)

                file.close()

    def animate2D(self, i, field):
        if i%self.plotInterval == 0:

            # adaptive limits y-axis
            limits = np.empty((4,3))

            for j in range(3):
                limits[j,0] = np.amin(field.field[j][:,int(self.Ny/2)])
                limits[j,1] = np.amax(field.field[j][:,int(self.Ny/2)])

            limits[3,0] = np.amin(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2][:,int(self.Ny/2)]))

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

            self.im0.set_array(field.field[0].T)
            self.im1.set_array(field.field[1].T)
            self.im2.set_array(field.field[2].T)
            self.im3.set_array(DowsonHigginson(self.rho0, self.P0).isoT_pressure(field.field[2].T))

            # lower = np.amin(field.field[comp])
            # upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            # lower = -50.
            # upper = 0.0

            # v1 = np.linspace(lower, upper, 11, endpoint=True)
            # self.cbar.set_ticks(v1)
            # self.cbar.draw_all()
            # self.time_text.set_text('time = %.6f ms' % (i*self.dt *1e3))

        self.solve(i)

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
