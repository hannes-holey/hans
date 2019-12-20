#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class EulerCentral:

    def __init__(self, options, geometry, numerics, material, BC, EOS):

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

        self.mu = float(material['mu'])

        self.U = float(BC['U'])
        self.V = float(BC['V'])
        self.P0 = float(BC['P0'])
        self.rho0 = float(BC['rho0'])
        self.temp = float(BC['temp'])

        self.eos = EOS[int(self.temp)]

        self.dimless = 6.*self.mu * self.U *self.Lx/self.h2**2

        from field.field import scalarField
        from field.field import vectorField

                # Gap height
        self.height = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.height.fromFunctionXY(self.heightLinear)

        # Pressure
        self.press = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        # self.press.fromFunctionXY(self.pressLinear)

        # Density distribution
        self.rho = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.rho.normal(self.rho0, 0.)
        # self.rho.fromFunctionField(self.DowsHigg_isoT_rho, self.press.field[0])

        self.press.fromFunctionField(self.DowsHigg_isoT_P, self.rho.field[0])

        # Pressure gradient
        self.pGrad = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Flux field
        self.flux = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxDiv = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.vel = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Divergence of stress tensor
        self.stressDiv = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        if self.writeOutput == True:
            self.tagO = 0
            while 'out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat' in os.listdir('./output'):
                self.tagO += 1
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%14s \t %14s \n" % ('mass', 'vmax'))
        if self.writeField == True:
            self.tagF = 0
            while 'field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat' in os.listdir('./output'):
                self.tagF += 1

        self.plot(self.press, 0)


    def solve(self):

        self.rho.setDirichletXe(self.DowsHigg_isoT_rho(self.P0))
        self.rho.setDirichletXw(self.DowsHigg_isoT_rho(self.P0))

        self.press.fromFunctionField(self.DowsHigg_isoT_P, self.rho.field[0])

        # self.pGrad.computeGrad(self.press)

        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny
        self.pGrad.field[0], self.pGrad.field[1] = np.gradient(self.press.field[0], dx,dy, edge_order=2)

        self.stressDiv.fromField(self.pGrad)
        self.stressDiv.addFluxContribution(self.stressDivNewtonian, \
                self.height.field[0], self.flux.field[0], self.flux.field[1], self.rho.field[0])

        self.flux.updateFlux(self.stressDiv, self.dt)
        self.flux.setPeriodicY()
        #self.flux.setPeriodicX()

        self.fluxDiv.computeDiv(self.flux)
        self.rho.updateDens(self.fluxDiv, self.dt)

        self.vel.field[0] = self.flux.field[0]/self.rho.field[0]

        if self.writeOutput == True:
            mass = np.sum(self.rho.field[0] * self.height.field[0])*self.Lx*self.Ly
            vmax = np.amax(np.abs(self.vel.field[0]))
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%.8e \t %.8e \n" % (mass, vmax))

    def plot(self, field, comp):

        if self.plot_dim == 1:
            fig, self.ax1 = plt.subplots(1)
            x = np.linspace(0,self.Lx,self.Nx, endpoint=True)
            self.line, = self.ax1.plot(x, field.field[0][:,int(self.Ny/2)])
            self.ax1.plot(x, self.P_analytical(x) * 1e-3  + self.P0, '-')
            self.time_text = self.ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
            ani = animation.FuncAnimation(fig, self.animate1D, self.maxIt, fargs=(field, comp), interval=1 ,repeat=False)
            self.ax1.grid()

        elif self.plot_dim == 2:
            fig, self.ax1 = plt.subplots(1)
            # lower = np.amin(field.field[comp][:,:])
            # upper = np.amax(field.field[comp][:,:])
            self.im = self.ax1.imshow(field.field[0], interpolation='nearest')
            self.time_text = self.ax1.text(0.05, 1.05,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
            # v1 = np.linspace(lower, upper, 20, endpoint=True)
            self.cbar = plt.colorbar(self.im, ax = self.ax1)
            ani = animation.FuncAnimation(fig, self.animate2D, self.maxIt, fargs=(field, comp), interval=1 ,repeat=False)

        if self.save_ani == True:
            i = 0
            while str(self.name) + '_' + str(i).zfill(2) + '.mp4' in os.listdir('./output'):
                i += 1
            ani.save('./output/'+ self.name + '_' + str(i).zfill(2) + '.mp4',fps=30)
        else:
            plt.show()

    def animate1D(self, i, field, comp):
        if i%self.plotInterval == 0:
            lower = np.amin(field.field[comp][:,int(self.Ny/2)])
            upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            if upper == lower:
                vspace = 0.5*upper
            else:
                vspace = 0.1*(upper - lower)
            self.time_text.set_text('time = %.3f µs' % (i*self.dt * 1e6) )
            self.line.set_ydata(field.field[comp][:,int(self.Ny/2)])
            self.ax1.set_ylim(lower - vspace , upper + vspace)

            if self.writeField == True:
                out = np.vstack((field.xx[:,int(self.Ny/2)],field.field[comp][:,int(self.Ny/2)])).T
                with open('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat', "a+") as f:
                    np.savetxt(f, out, header='step' + str(i))
                #     f.write(str(i) + '\n' + str(field.field[comp]) + '\n')
        self.solve()

    def animate2D(self, i, field, comp):
        if i%self.plotInterval == 0:
            lower = np.amin(field.field[comp])
            upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            self.im.set_clim(vmin=lower,vmax=upper)
            # v1 = np.linspace(lower, upper, 11, endpoint=True)
            # self.cbar.set_ticks(v1)
            # self.cbar.draw_all()
            self.time_text.set_text('time = %.6f ms' % (i*self.dt *1e3))
            self.im.set_array(field.field[comp])
        self.solve()

    def heightLinear(self, x , y):
        "Linear height profile"
        sx = (self.h2 - self.h1)/self.Lx
        sy = 0.
        return self.h1 + sx * x + sy * y

    def stressDivNewtonian(self, h, j, rho):
        "compute flux-dependent term in divergence of stress tensor"
        return 12. * self.mu/(h**2)*(j/rho - self.U/2.)

    def pressLinear(self, x , y):
        "Linear height profile"
        p2 = 101325.
        p1 = 202650.
        sx = (p2 - p1)/self.Lx
        sy = 0.
        return p1 + sx * x + sy * y

    def EqOfState(self, rho):
        "Equation of state"
        p = np.poly1d(self.eos)
        return p(rho)*1.e6


    def EqOfState_lin(self, rho):
        p1 = 0.
        p2 = 119.e6
        rho1 = 0
        rho2 = 1000
        return p1 + (p2 - p1)/(rho2 - rho1) * (rho -rho1)

    def invEqofState(self, p):
        p1 = 0.
        p2 = 119.e6
        rho1 = 0
        rho2 = 1000
        return  rho1 + (rho2 - rho1)/(p2 - p1) * (p - p1)

    def Height(self, x, alpha):
        return alpha + (1. - alpha)*x

    def P_analytical(self, x):
        alpha = self.h1/self.h2
        tmp = alpha/(1-alpha**2) * (1./self.Height(x/self.Lx,alpha)**2 - 1./alpha**2) \
        - 1./(1. - alpha) * (1./self.Height(x/self.Lx, alpha) - 1./alpha)

        return tmp

    def DowsHigg_isoT_P(self, rho):
        B1 = 6.6009e-10
        B2 = 2.8225e-9
        P0= 101325.
        return (rho - self.rho0)/(B1*self.rho0 - B2*(rho - self.rho0)) + P0

    def DowsHigg_isoT_rho(self, P):
        B1 = 6.6009e-10
        B2 = 2.8225e-9
        P0= 101325.
        return self.rho0 * (1. + B1*(P - P0 /(1.+B2*(P-P0))))
