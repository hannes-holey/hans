#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import h5py

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

        self.sx = (self.h2 -self.h1)/self.Lx
        self.sy = 0.

        self.Nx = int(numerics['Nx'])
        self.Ny = int(numerics['Ny'])
        self.dt = float(numerics['dt'])
        self.freq = int(numerics['freq'])

        self.mu = float(material['mu'])

        # PR = 0.0
        # self.lam = 2.*self.mu*PR/(1.-2. * PR)

        # Stokes assumption
        self.lam = -2./3. * self.mu

        self.U = float(BC['U'])
        self.V = float(BC['V'])
        self.P0 = float(BC['P0'])
        self.rho0 = float(BC['rho0'])
        self.temp = float(BC['temp'])

        self.eos = EOS[int(self.temp)]

        self.dimless = 6.*self.mu * self.U *self.Lx/self.h2**2

        from field.field import scalarField
        from field.field import vectorField
        from field.field import tensorField

        # Gap height
        self.height = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.height.fromFunctionXY(self.heightLinear)

        # Pressure

        self.press = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        # self.press.fromFunctionXY(self.pressLinear)

        # Density distribution
        self.rho = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.rho.normal(self.rho0, 0.)

        self.rho.setDirichletXe(self.DowsHigg_isoT_rho(self.P0))

        if self.name == 'inclined':
            self.rho.setDirichletXw(self.DowsHigg_isoT_rho(self.P0))
        elif self.name == 'poiseuille':
            self.rho.setDirichletXw(self.DowsHigg_isoT_rho(2.*self.P0))

        self.rho_old = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.rho_old.field[0] = self.rho.field[0]

        self.press.fromFunctionField(self.DowsHigg_isoT_P, self.rho.field[0])

        self.stress = tensorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Pressure gradient
        self.pGrad = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Flux field
        self.flux = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.flux_old = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.fluxDiv = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxDiv_helper = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.vel = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.alp = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Divergence of stress tensor
        self.stressDiv = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.stressDiv_helper = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)


        if self.writeOutput == True:
            self.tagO = 0
            while 'out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat' in os.listdir('./output'):
                self.tagO += 1
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%14s \t %14s \t %14s \t %14s \n" % ('time', 'mass', 'vmax', 'jXmax'))
        if self.writeField == True:
            self.tagF = 0
            while 'field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat' in os.listdir('./output'):
                self.tagF += 1

        self.plot(self.flux, 0)

    def solve(self, i):
        # BCs
        self.flux.setPeriodicY()
        self.rho.setDirichletXe(self.DowsHigg_isoT_rho(self.P0))

        if self.name == 'inclined':
            self.rho.setDirichletXw(self.DowsHigg_isoT_rho(self.P0))
        elif self.name == 'poiseuille':
            self.rho.setDirichletXw(self.DowsHigg_isoT_rho(2.*self.P0))

        self.press.fromFunctionField(self.DowsHigg_isoT_P, self.rho.field[0])

        self.stress.getStressNewtonian(self.press.field[0], self.height.field[0], self.rho.field[0], \
                self.flux.field[0], self.flux.field[1], self.mu, self.lam, self.sx, self.sy, self.U, self.V)

        # Central differences
        self.stressDiv.computeDiv_CD(self.stress)
        self.fluxDiv.computeDiv_CD(self.flux)

        # self.stressDiv_helper.computeHelper_LW(self.stress)
        # self.fluxDiv_helper.computeHelper_LW(self.flux)

        # Upwind backward
        # self.stressDiv.computeDiv_BW(self.stress)
        # self.fluxDiv.computeDiv_BW(self.flux)

        # Upwind forward
        # self.stressDiv_helper.computeDiv_FW(self.stress)
        # self.fluxDiv_helper.computeDiv_FW(self.flux)

        # Lax-Friedrichs
        self.flux.updateFlux_LF(self.stressDiv, self.dt)
        self.rho.updateDens_LF(self.fluxDiv, self.dt)

        # Lax-Wendroff
        # self.flux.updateFlux_LW(self.stressDiv, self.stressDiv_helper, self.dt)
        # self.rho.updateDens_LW(self.fluxDiv, self.fluxDiv_helper, self.dt)

        # explicit Euler
        # self.flux.updateFlux(self.stressDiv, self.dt)
        # self.rho.updateDens(self.fluxDiv, self.dt)

        # Leapfrog
        # self.flux.field[0], self.flux.field[1] = self.flux_old.updateFlux_leapfrog(self.stressDiv, self.flux, self.dt)
        # self.rho.field[0] = self.rho_old.updateDens_leapfrog(self.fluxDiv, self.rho, self.dt)

        # MacCormack
        # self.rho_old = self.rho
        #
        # self.flux.updateFlux_MC(self.stressDiv)
        # self.flux.updateFlux_MC(self.stressDiv_helper)
        # self.rho.updateFlux_MC(self.fluxDiv)
        # self.rho.updateFlux_MC(self.fluxDiv_helper)


        self.vel.field[0] = self.flux.field[0]/self.rho.field[0]
        self.alp.field[0] =  6./self.height.field[0]**2 * (self.flux.field[0]/self.rho.field[0] - self.U/2.)

        if self.writeOutput == True:
            mass = np.sum(self.rho.field[0] * self.height.field[0])*self.Lx*self.Ly
            vmax = np.amax(np.abs(self.vel.field[0]))
            maxFlux_X = np.amax(self.flux.field[0])
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%.8e \t %.8e \t %.8e \t %.8e \n" % (i*self.dt, mass, vmax, maxFlux_X))

    def plot(self, field, comp):

        if self.plot_dim == 1:
            fig, self.ax1 = plt.subplots(1)
            x = np.linspace(0, self.Lx, self.Nx, endpoint=True)
            # z = np.linspace(0., self.h1, 10)
            # x = np.linspace(0, self.h1, 10, endpoint=True)
            self.line, = self.ax1.plot(x, field.field[comp][:,int(self.Ny/2)])
            #self.ax1.set_xlabel('distance (mm)')
            #self.ax1.set_ylabel('pressure (atm)')
            #self.ax1.plot(x, -4e-6*self.P_analytical(x)*self.dimless  + self.P0, '-')
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

        elif self.plot_dim == 3:
            fig, self.ax1 = plt.subplots(1)
            # x = np.linspace(0, self.Lx, self.Nx, endpoint=True)
            z = np.linspace(0., self.height.field[0][-1, int(self.Ny/2)], 50)
            # x = np.linspace(0, self.h1, 10, endpoint=True)
            self.line, = self.ax1.plot((self.U/self.height.field[0][-1, int(self.Ny/2)] + field.field[comp][-1, int(self.Ny/2)]*z)*(self.height.field[0][-1, int(self.Ny/2)] - z),z)
            # self.line, = self.ax1.plot(x, field.field[comp])
            #self.ax1.plot(x, -self.P_analytical(x)*self.dimless  + self.P0, '-')
            self.time_text = self.ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=self.ax1.transAxes)
            ani = animation.FuncAnimation(fig, self.animate1D_z, self.maxIt, fargs=(field, comp), interval=1 ,repeat=False)
            self.ax1.grid()

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
            # lower = 1e5/101325
            # lower = 40.
            # lower = 101325
            #lower = 0.
            upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            # upper = 1.08e5/101325
            # upper = 50.
            # upper = 101500
            #upper = 1500.
            if upper == lower:
                vspace = 0.5*upper
            else:
                vspace = 0.1*(upper - lower)
            self.time_text.set_text('step = %1d' % (i) )
            # self.time_text.set_text('time = %.3f µs' % (i*self.dt * 1e6) )

            # z = np.linspace(0., self.h1, 10)
            self.line.set_ydata(field.field[comp][:,int(self.Ny/2)])

            # self.line.set_ydata(field.field[comp])

            self.ax1.set_ylim(lower - vspace , upper + vspace)

            if self.writeField == True:

                out = np.vstack((field.xx[:,int(self.Ny/2)],field.field[comp][:,int(self.Ny/2)])).T


                with open('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat', "a+") as f:
                    np.savetxt(f, out, header='step' + str(i))


                #     f.write(str(i) + '\n' + str(field.field[comp]) + '\n')
        self.solve(i)

    def animate1D_z(self, i, field, comp):
        if i%self.plotInterval == 0:
            # adaptive bounds
            # lower = np.amin(field.field[comp][0, int(self.Ny/2)]*self.h1**2/4.)
            # upper = np.amax(field.field[comp][:, int(self.Ny/2)]*self.h1**2/4.)

            # inclined
            lower = -0.01
            upper = 0.1

            # poiseuille
            #lower = -0.000
            #upper = 2e-3

            if upper == lower:
                vspace = 0.5*upper
            else:
                vspace = 0.1*(upper - lower)
            self.time_text.set_text('step = %.1f' % (i) )
            # self.time_text.set_text('time = %.3f µs' % (i*self.dt * 1e6) )

            z = np.linspace(0., self.height.field[0][0, int(self.Ny/2)], 50)

            # right
            #self.line.set_xdata((self.U/self.height.field[0][-1, int(self.Ny/2)] + field.field[comp][-1, int(self.Ny/2)]*z)*(self.height.field[0][-1, int(self.Ny/2)] - z))
            # left
            self.line.set_xdata((self.U/self.height.field[0][0, int(self.Ny/2)] + field.field[comp][0, int(self.Ny/2)]*z)*(self.height.field[0][0, int(self.Ny/2)] - z))
            # middle
            # self.line.set_xdata((self.U/self.height.field[0][int(self.Nx/2), int(self.Ny/2)] + field.field[comp][int(self.Nx/2), int(self.Ny/2)]*z)*(self.height.field[0][int(self.Nx/2), int(self.Ny/2)] - z))


            self.ax1.set_xlim(lower - vspace , upper + vspace)

            if self.writeField == True:
                out = np.vstack((field.xx[:,int(self.Ny/2)],field.field[comp][:,int(self.velNy/2)])).T
                with open('./output/field_' + str(self.name) + '_' + str(self.tagF).zfill(2) + '.dat', "a+") as f:
                    np.savetxt(f, out, header='step' + str(i))
                #     f.write(str(i) + '\n' + str(field.field[comp]) + '\n')
        self.solve(i)

    def animate2D(self, i, field, comp):
        if i%self.plotInterval == 0:
            lower = np.amin(field.field[comp])
            # lower = 0.
            # upper = 1.
            upper = np.amax(field.field[comp][:,int(self.Ny/2)])
            self.im.set_clim(vmin=lower,vmax=upper)
            # v1 = np.linspace(lower, upper, 11, endpoint=True)
            # self.cbar.set_ticks(v1)
            # self.cbar.draw_all()
            self.time_text.set_text('time = %.6f ms' % (i*self.dt *1e3))
            self.im.set_array(field.field[comp])
        self.solve(i)

    def heightLinear(self, x , y):
        "Linear height profile"
        sx = (self.h2 - self.h1)/(2.*self.Lx)
        sy = 0.


        # if x < self.Lx/2:
        #     h = self.h1 + sx * x + sy * y
        # else:
        #     h = self.h1 + (self.h2 - self.h1)/2. - sx * x + sy * y
        # return h
        return self.h1 + sx * x + sy * y

    def stressDivNewtonian(self, h, j, rho):
        "compute flux-dependent term in divergence of stress tensor"
        return 12. * self.mu/(h**2)*(j/rho - self.U/2.)

    def stressDivNewtonian_lower(self, h, jx, jy, rho):
        v1 = (4.*self.U*(self.sx**2*(self.mu + 0.5*self.lam)+1.5*self.mu)*rho-12.*jx*((self.mu+0.5*self.lam)*self.sx**2+self.mu))/(rho*h**2)
        #v2 = (2.*self.mu*(self.V*(self.sx**2+3.)*rho - 3.*jy*self.sx**2 - 6.*jy))/(rho*h**2)
        #v1 = (2.*(self.mu + self.lam)*(self.U*rho-3.*jx)*self.sx**2 + 6.*self.mu*(self.U*rho-2.*jx))/(rho*h**2)
        v2 = (2.*self.mu*(self.V*(self.sx**2+3.)*rho - 3.*jy*self.sx**2 - 6.*jy))/(rho*h**2)
        v3= (-2.*(self.mu + self.lam)*(self.U*rho - 3.*jx)*self.sx)/(rho*h**2)

        return v1, v2, v3

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

class LaxFriedrichs:

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

        self.sx = (self.h2 -self.h1)/self.Lx
        self.sy = 0.

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

        # self.eos = EOS[int(self.temp)]

        self.dimless = 6.*self.mu * self.U *self.Lx/self.h2**2

        from field.field import scalarField
        from field.field import vectorField
        from field.field import tensorField

        # Gap height
        self.height = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.height.fromFunctionXY(self.heightLinear)

        self.q = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.q.field[2][:,:] = self.rho0 * np.ones(shape=(self.Nx, self.Ny))

        if self.name == 'inclined':
            self.q.field[2][0,:] = self.DowsHigg_isoT_rho(self.P0)
            self.q.field[2][-1,:] = self.DowsHigg_isoT_rho(self.P0)
        # if self.name == 'inclined':
        #     self.q.field[2][-1,:] = self.DowsHigg_isoT_rho(self.P0)
        #     self.q.field[2][0,:] = self.DowsHigg_isoT_rho(self.P0)
        elif self.name == 'poiseuille':
            self.q.field[2][-1,:] = self.DowsHigg_isoT_rho(self.P0)
            self.q.field[2][0,:] = self.DowsHigg_isoT_rho(2.*self.P0)


        self.fluxX_E = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxX_W = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxY_N = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)
        self.fluxY_S = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        self.rhs = vectorField(self.Nx, self.Ny, self.Lx, self.Ly)

        # Pressure
        self.press = scalarField(self.Nx, self.Ny, self.Lx, self.Ly)
        #self.press.fromFunctionField(self.DowsHigg_isoT_P, self.q.field[2])

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

        self.press.fromFunctionField(self.DowsHigg_isoT_P, self.q.field[2])

        if i%self.freq == 0:
        # if i == 1:
            self.stress.getStressNewtonian_avg(self.press.field[0], self.height.field[0], self.q.field[2], \
                    self.q.field[0], self.q.field[1], self.mu, self.lam, self.sx, self.sy, self.U, self.V)
            self.stress.addNoise(self.s)


        self.fluxX_E.fluxX_LF(self.stress, self.q, self.dt, -1)
        self.fluxX_W.fluxX_LF(self.stress, self.q, self.dt,  1)

        self.fluxY_N.fluxY_LF(self.stress, self.q, self.dt, -1)
        self.fluxY_S.fluxY_LF(self.stress, self.q, self.dt,  1)

        self.rhs.compRHS(self.fluxX_E, self.fluxX_W, self.fluxY_N, self.fluxY_S)
        self.rhs.addStress_wall(self.height.field[0], self.q.field[2], self.q.field[0], self.q.field[1], self.mu, self.U, self.V)

        # self.rhs.computeRHS(self.stress, self.q, self.dt)

        self.q.updateLF(self.rhs, self.dt)

        # Central differences
        # self.rhs.compute_RHS_1(self.stress, self.q)

        # Central Differences for flux (3rd eq)
        # self.rhs.computeFluxDiv_np(self.q)


        # Central differeces press (1st /2nd eq)
        # self.rhs.getPgrad_np(self.press)
        # analytical divergence of viscous stress tensor
        # self.rhs.getStressDiv_avg(self.height.field[0], self.q.field[2], self.q.field[0], self.q.field[1], \
                                # self.mu, self.lam, self.sx, self.sy, self.U, self.V)

        # Time update Lax-Friedrichs
        # self.q.update_LF(self.rhs, self.dt)
        #self.q.update_LF_full(self.rhs, self.dt)
        # self.q.updateLF_new(self.rhs, self.dt)

        # Periodic BCs (y)
        self.q.field[0][1:-1,0] = self.q.field[0][1:-1,-1]
        self.q.field[1][1:-1,0] = self.q.field[1][1:-1,-1]
        self.q.field[2][1:-1,0] = self.q.field[2][1:-1,-1]

        # self.q.field[0][0,:] = self.q.field[0][1,:]
        # self.q.field[0][-1,:] = self.q.field[0][-2,:]

        # Density BCs
        if self.name == 'inclined':
            # pass
            # self.q.field[0][0,:] = self.q.field[0][1,:]
            # self.q.field[0][-1,:] = self.q.field[0][-2,:]
            # self.q.field[1][0,:] = self.q.field[1][1,:]
            # self.q.field[1][-1,:] = self.q.field[1][-2,:]
            self.q.field[2][0,:]  = self.DowsHigg_isoT_rho(self.P0)
            self.q.field[2][-1,:] = self.DowsHigg_isoT_rho(self.P0)
            # self.q.field[2][0:3,:] = self.DowsHigg_isoT_rho(self.P0)
            # self.q.field[2][-3:,:] = self.DowsHigg_isoT_rho(self.P0)
        elif self.name == 'poiseuille':
            pass
            # self.q.field[2][0,:] = self.DowsHigg_isoT_rho(2.*self.P0)
            # self.q.field[2][-1,:] = self.DowsHigg_isoT_rho(self.P0)
            # self.q.field[2][0:2,:] = self.DowsHigg_isoT_rho(2.*self.P0)
            # self.q.field[2][-2:,:] = self.DowsHigg_isoT_rho(self.P0)
        # ----------------------------------------------------------------------

        self.vel.field[0] = self.q.field[1]/self.q.field[2]
        self.alp.field[0] =  6./self.height.field[0]**2 * (self.q.field[0]/self.q.field[2] - self.U/2.)

        self.mass = np.sum(self.q.field[2] * self.height.field[0])*self.Lx*self.Ly

        if self.writeOutput == True:
            mass = np.sum(self.q.field[2] * self.height.field[0])*self.Lx*self.Ly
            vmax = np.amax(np.abs(self.vel.field[0]))
            maxFlux_X = np.amax(self.q.field[0])
            with open('./output/out_' + str(self.name) + '_' + str(self.tagO).zfill(2) + '.dat', "a+") as f:
                f.write("%.8e \t %.8e \t %.8e \t %.8e \n" % (i*self.dt, mass, vmax, maxFlux_X))

    def plot(self, field):

        if self.plot_dim == 1:
            self.fig, self.ax1 = plt.subplots(2,2, figsize = (14,9), sharex=True)

            x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

            self.line0, = self.ax1[0,0].plot(x, field.field[0][:,int(self.Ny/2)])
            self.line1, = self.ax1[0,1].plot(x, field.field[1][:,int(self.Ny/2)])
            self.line2, = self.ax1[1,0].plot(x, field.field[2][:,int(self.Ny/2)])
            self.line3, = self.ax1[1,1].plot(x, self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))

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

            limits[3,0] = np.amin(self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))

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
            self.line3.set_ydata(self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))

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

            limits[3,0] = np.amin(self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))
            limits[3,1] = np.amax(self.DowsHigg_isoT_P(field.field[2][:,int(self.Ny/2)]))

            # manual limits y-axis
            limits[0,0] = 40.
            limits[0,1] = 45.
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
            self.im3.set_array(self.DowsHigg_isoT_P(field.field[2]).T)

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

    def heightLinear(self, x , y):
        "Linear height profile"
        sx = (self.h2 - self.h1)/(self.Lx)
        sy = 0.

        return self.h1 + sx * x + sy * y

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
        return self.rho0 * (1. + B1*(P - P0) /(1.+B2*(P-P0)))
