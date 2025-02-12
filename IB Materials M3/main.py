import sys
import numpy as np

from ase import Atoms
from ase.build import bulk
import Morse

from ase.eos import EquationOfState
from ase.units import kJ
from ase.units import eV, GPa

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from colour import Color

def main():

    D12 = Deliverables12()
    D12.del1main()
    D12.plotter()
    D12.unit_test()

    D3 = Deliverables34()
    D3.del3main()
    D3.plotter()

class Deliverables12:

    def __init__(self):

        self.min = 2.25
        self.max = 4

        self.ds = None
        self.sparse_ds = None

        self.potfunc = None
        self.forcefunc = None
        self.PE_poly = None
        self.approx = None

        return

    def del1main(self):

        self.ds = np.linspace(self.min, self.max, 50)
        self.sparse_ds = np.linspace(self.min, self.max, 10)

        self.potfunc = np.vectorize(self.get_pots)
        self.forcefunc = np.vectorize(self.get_forces)
        self.PE_poly = np.polynomial.Polynomial.fit(self.ds, self.potfunc(self.ds), 10)
        self.approx = lambda e, ds: -(self.potfunc(ds + e) - self.potfunc(ds)) / e

        return

    def get_pots(self,d):

        a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., d)])
        a.calc = Morse.MorsePotential()
        return a.get_potential_energy()

    def get_forces(self,d):

        a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., d)])
        a.calc = Morse.MorsePotential()
        return a.get_forces()[1][2]

    def unit_test(self):

        """
        Calculates the force between atoms in two different ways:
            1. use the built-in a.get_forces()
            2. use a.get_potential_energy(), fit a polynomial,
            compute negative derivative of polynomial (with numpy.Polynomial)

        Assert the two are within an acceptable margin of error - here, this is 0.1%
        """

        assert (-10**-3 < np.average(np.divide(
                             self.PE_poly.deriv(m=1)(self.ds),
                             self.forcefunc(self.ds))) + 1
                < 10**-3)

    def plotter(self):

        fig = plt.figure(layout="constrained")
        fig.suptitle('Deliverables 1/2')
        gs = GridSpec(5, 4, figure=fig)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:5])
        ax3 = fig.add_subplot(gs[2:5, 0:2])
        ax4 = fig.add_subplot(gs[2:5, 2:5])
        axs = [ax1, ax2, ax3]

        axs[0].plot([self.min, self.max],
                           [0,        0],
                            linestyle='dotted', color='red')

        axs[0].scatter(self.sparse_ds, self.potfunc(self.sparse_ds), marker='X')
        axs[0].plot(self.ds, self.PE_poly(self.ds), linestyle='--', color='black',zorder=0)
        axs[0].set_xlabel('Distance / Angstroms')
        axs[0].set_ylabel('Potential Energy / eV')

        axs[1].plot([self.min,     self.max],
                           [0,        0],
                            linestyle='dotted', color='red')
        axs[1].scatter(self.sparse_ds, self.forcefunc(self.sparse_ds),marker='X')
        axs[1].plot(self.ds, self.forcefunc(self.ds), linestyle='--', color='black',zorder=0)
        axs[1].set_xlabel('Distance / Angstroms')
        axs[1].set_ylabel('Force / eV')


        n = 5
        colors = list(Color("red").range_to(Color("green"), n))
        for i,e in enumerate(np.linspace(0.5,2, n)):
            axs[2].plot(self.ds[:25],self.approx(10**-e,self.ds[:25]),color=colors[i].hex,zorder = -i)
        axs[2].plot(self.ds[:25], self.forcefunc(self.ds[:25]), linestyle='--', color='black',zorder=0, linewidth=2)
        axs[2].set_xlabel('Distance / Angstroms')
        axs[2].set_ylabel('Force / eV')


        es = np.linspace(10,17, 100)
        errors = np.array([self.approx(10**-e,self.ds)[np.absolute((self.ds - 2.67)).argmin()] for i,e in enumerate(es)])

        ax4.plot(es,errors)
        ax4.set_xlabel('Negative power of 10 for eta')
        ax4.set_ylabel('Force value for d = 2.67A')
        fig.show()

        return


class Deliverables34:

    def __init__(self):

        self.min = 0.95
        self.max = 1.05

        self.strains = None
        self.volumes = None
        self.pressures = None
        self.pot_energies = None
        self.equilibrium_volume = None
        self.equimin = None

        pass

    def del3main(self):

        self.strains = np.linspace(self.min,self.max, 50)
        self.volumes = np.vectorize(self.get_volumes)(self.strains)
        self.pressures = np.vectorize(self.get_pressure)(self.strains)
        self.pot_energies = np.vectorize(self.get_potential_energies)(self.strains)

        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        cu.calc = Morse.MorsePotential()
        self.equilibrium_volume = cu.get_volume()

        self.equimin = np.polynomial.Polynomial.fit(self.volumes * 4, self.pot_energies,6).deriv().roots().real[2]**(1/3)

        self.get_bulk()
        self.get_shear_modulus()
        self.get_poisson()

        self.del5()

        return

    def del5(self):

        print(f"\nDeliverable 5: \n"
              f"\t1. Dislocations are the lighter green streaks through the materials\n"
              f"\t2. Dislocations propagate through the lattice until they reach a free surface, "
              f"\n\t\tat which point they make a kink/crack on the surface\n"
              f"\t3. At the notch, the deformations propagate from an imperfection, "
              f"\n\t\tand form a crack that grows as the animation goes on. "
              f"\n\t\tEventually would cause fast fracture ")

    def get_shear_modulus(self):
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        cu.calc = Morse.MorsePotential()
        cu.cell[0][1] = 0.01*cu.cell[0][0]
        shear_modulus = cu.get_stress(voigt=False)[0][1]/(0.01)
        shear_modulus /= 2 # engineering shear
        print(f'Shear Modulus: {shear_modulus / GPa} GPa')

        return

    def get_poisson(self):

        e0 = 0.01
        trial_ratios = np.linspace(0.15,0.25, 30)

        yzstresses = np.zeros_like(trial_ratios)

        for i,tr in enumerate(trial_ratios):
            cu = bulk("Cu", "fcc", a=self.equimin, cubic=True)
            cu.calc = Morse.MorsePotential()

            cu.cell[0][0] *= (1 + e0)
            cu.cell[1][1] *= (1 - e0 * tr)
            cu.cell[2][2] *= (1 - e0 * tr)

            yzstresses[i] = cu.get_stress(voigt=False)[1][1]

        poisson_P = np.polynomial.Polynomial.fit(
                     trial_ratios, yzstresses, 1)

        print(f'Poisson Ratio: {poisson_P.roots()[0]}')

    def get_bulk(self):

        pressure_poly = np.polynomial.Polynomial.fit(self.volumes, self.pressures, 5).deriv()
        vols2 = np.linspace(np.min(self.volumes),np.max(self.volumes),500)
        zeroindex = np.argmin(np.absolute(vols2 - (self.equimin**3) / 4))
        pressure_grad = pressure_poly(vols2[zeroindex])

        print(f'Bulk Modulus: {- pressure_grad * vols2[zeroindex] / GPa} GPa')


    def get_volumes(self, strain):

        local_cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        local_cu.calc = Morse.MorsePotential()
        cell = local_cu.get_cell()

        cell *= strain
        local_cu.set_cell(cell, scale_atoms=True)

        return local_cu.get_volume() / 4

    def get_pressure(self, strain):
        local_cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        local_cu.calc = Morse.MorsePotential()
        cell = local_cu.get_cell()

        cell *= strain
        local_cu.set_cell(cell, scale_atoms=True)

        return -np.trace(local_cu.get_stress(voigt=False)) / 3

    def get_potential_energies(self, strain):

        local_cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        local_cu.calc = Morse.MorsePotential()

        cell = local_cu.get_cell()
        cell *= strain
        local_cu.set_cell(cell, scale_atoms=True)

        return local_cu.get_potential_energy() / local_cu.get_global_number_of_atoms()

    def plotter(self):

        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Deliverables 3/4')

        axs[0].plot([self.equilibrium_volume /4, self.equilibrium_volume /4],[np.min(self.pressures) /GPa,np.max(self.pressures) /GPa], linestyle='--', color='black')
        axs[0].scatter(self.volumes, self.pressures / GPa,zorder=2)
        axs[0].set_ylabel('Pressure / GPa')
        axs[0].set_xlabel('Volume / A^3')

        axs[1].plot([self.equilibrium_volume /4, self.equilibrium_volume /4],[np.min(self.pot_energies),np.max(self.pot_energies)], linestyle='--', color='black')
        axs[1].scatter(self.volumes, self.pot_energies,zorder = 2)
        axs[1].set_ylabel('Potential Energy / eV')
        axs[1].set_xlabel('Volume / A^3')

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()


