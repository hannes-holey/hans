import numpy as np
import subprocess
import scipy.constants as sci
from ase.lattice.cubic import FaceCenteredCubic


# from argparse import ArgumentParser


# def get_parser():

#     parser = ArgumentParser()
#     parser.add_argument('-mol', '--molecule', type=str, default='pentane',
#                         choices=['pentane', 'decane', 'hexadecane'])
#     parser.add_argument('-n', '--size', type=int, default=3)
#     parser.add_argument('-g', '--gap-height', type=float, default=30.)
#     parser.add_argument('-d', '--density', type=float, default=.7)
#     parser.add_argument('-jx', '--fluxX', type=float, default=.0)
#     parser.add_argument('-jy', '--fluxY', type=float, default=.0)
#     parser.add_argument('-U', '--wall-velocity', type=float, default=20.)
#     parser.add_argument('-T', '--temperature', type=float, default=300.)
#     parser.add_argument('-s', '--solid', type=str, default='Au')
#     parser.add_argument('-b', '--build', action='store_true', default=False)
#     parser.add_argument('-C', '--couette', action='store_true', default=False)

#     parser.add_argument('-dt', '--timestep', type=float, default=1.)
#     parser.add_argument('--Ninit', type=int, default=50_000)
#     parser.add_argument('--Nsteady', type=int, default=100_000)
#     parser.add_argument('--Nsample', type=int, default=300_000)
#     parser.add_argument('-Ne', '--Nevery', type=int, default=10)
#     parser.add_argument('-Nr', '--Nrepeat', type=int, default=100)
#     parser.add_argument('-Nf', '--Nfreq', type=int, default=1000)
#     parser.add_argument('--nz', type=int, default=200)

#     parser.add_argument('-r', '--restart-file', type=str, default='restart.steady')

#     return parser


def get_molecule_grid(file, lx, ly, h, Nf):
    coords = []

    file = 'moltemplate_files/' + file

    with open(file, 'r') as f:
        for lines in f.readlines():
            line = lines.split()
            if len(line) > 1 and line[0].startswith('$atom:'):
                # atom_type full
                coord = [float(x) for x in line[4:7]]
                coords.append(coord)

    coords = np.array(coords)
    lim = coords.max(0) - coords.min(0)
    mx, my, mz = lim

    nxf = int(np.floor(lx/(2*mx)))
    nyf = int(np.floor(ly/(2*my)))

    if nxf == 0 or nyf == 0:
        raise RuntimeError("Molecule larger than specified box. Increase n!")

    if Nf % (nxf * nyf) == 0:
        nzf = int(Nf / nxf / nyf)
    else:
        nzf = Nf // (nxf * nyf) + 1

    mz = max(mz, 1.5)
    gap = max(2. * nzf * mz, h)

    return nxf, nyf, nzf, gap


def write_solid_data(coords, mass):

    out = "solid {\n\n"

    # Coordinates
    out += "\twrite(\"Data Atoms\") {\n\t\t"
    # data = [f"$atom:au_{i+1} $mol:. @atom:au " + "0.0 " + " ".join(coord)
    #         for i, coord in enumerate(coords)]

    data = [f"$atom:au_{i+1} $mol:. @atom:au " +
            f"0.0 {coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}"
            for i, coord in enumerate(coords)]

    out += "\n\t\t".join(data) + "\n\t}\n\n"

    # Masses
    out += "\twrite_once(\"Data Masses\") {\n\t\t@atom:au "
    out += f"{mass}"
    out += "\n\t}\n\n"

    # Pair coeffs
    # Heinz et al., J. Phys. Chem. C 112 2008
    out += "\twrite_once(\"In Settings\") {\n"
    out += "\t\tpair_coeff @atom:au @atom:au lj/cut 5.29 2.629\n"
    out += "\t\tgroup solid type @atom:au\n\t}\n"
    out += "}\n\n"

    return out


def write_boundary(lx, ly, lz):

    out = "write_once(\"Data Boundary\") {\n\t"
    out += f"0. {lx:.4f} xlo xhi\n\t"
    out += f"0. {ly:.4f} ylo yhi\n\t"
    out += f"0. {lz:.4f} zlo zhi\n"
    out += "}\n"

    return out


def write_slab(nx, ny, nz, ax, ay, az, name='solid', shift=0.):
    out = f"""
{name} = new solid [{nx}].move({ax}, 0, 0)
                [{ny}].move(0, {ay}, 0)
                [{nz}].move(0, 0, {az})
"""

    out += f"\n{name}[*][*][*].move(0., 0., {az/6.+shift})\n"

    return out


def write_fluid(file, Nf, Nx, Ny, Nz, Lx, Ly, h, zoffset):
    out = f"""
import {file}
"""

    # Nfluid = round(sci.N_A * density * lx * ly * gap * 1.e-24 / M)

    name = file.split('.')[0]
    ax = Lx / Nx
    ay = Ly / Ny
    az = h / Nz

    delta = abs(Nf - Nx * Ny * Nz)

    out += f"""
fluid = new {name} [{Nx}].move({ax}, 0.0, 0.0)
                 [{Ny}].move(0.0, {ay}, 0.0)
                 [{Nz}].move(0.0, 0.0, {az})

fluid[*][*][*].move(0, 0, {zoffset})
"""
    i = 0
    diff = 0
    while diff < delta:
        out += f"delete fluid[0-{min(Nx, delta-diff-1)}][{i}][0]\n"
        i += 1
        diff += Nx

    return out


def write_settings(args, offset):

    density = args.get("density")
    wall_velocity = args.get("vWall")
    U = wall_velocity * 1e-5  # m/s to A/fs

    # Couette flow
    couette = args.get("couette", False)
    #
    if couette:
        jx_SI = density * wall_velocity / 2. * 1e3  # kg / m^2 s
        jx_real = jx_SI * sci.N_A * 1e-32  # g/mol/A^2/fs
        jy_real = 0.
    else:
        jx_real = args.get("fluxX")
        jy_real = args.get("fluxY")

    timestep = args.get("timestep", 1.)
    Ninit = args.get("Ninit", 50_000)
    Nsteady = args.get("Nsteady", 100_000)  # should depend on sliding velocity and size
    Nsample = args.get("Nsample", 300_000)
    temperature = args.get("temperature", 300.)

    nz = args.get("nz", 200)
    Nevery = args.get("Nevery", 10)
    Nrepeat = args.get("Nrepeat", 100)
    Nfreq = args.get("Nfreq", 1000)
    dumpfreq = args.get("Nfreq", 10_000)

    out = "\nwrite_once(\"In Settings\"){"
    out += f"""

    variable        offset equal {offset}  # mismatch between initial and target gap

    variable        dt equal {timestep}
    variable        Ninit equal {Ninit}
    variable        Nsteady equal {Nsteady}
    variable        Nsample equal {Nsample}

    variable        input_fluxX equal {jx_real}
    variable        input_fluxY equal {jy_real}
    variable        input_temp equal {temperature} # K
    variable        vWall equal {U} # A/fs

    # sampling // spatial
    variable        nbinz index {nz}

    # sampling // temporal
    variable        Nevery equal {Nevery}
    variable        Nrepeat equal {Nrepeat}
    variable        Nfreq equal {Nfreq}

    variable        dumpfreq equal {dumpfreq}


    include         static/in.settings.lmp

"""
    out += "}\n"

    return out


def write_run():

    # TODO: option to restart simulation

    out = """
write_once("In Run"){

    include static/in.run.equil.lmp
    include static/in.run.steady.lmp
    include static/in.run.sample.lmp

}
"""

    return out


def get_mass_alkane(name):

    molecules = {'pentane': [3, 2, 0],
                 'decane': [8, 2, 0],
                 'hexadecane': [14, 2, 0], }

    nCH2, nCH3, nCH4 = molecules[name]

    mCH2 = 14.1707
    mCH3 = 15.2507
    mCH4 = 16.3307

    return nCH2 * mCH2 + nCH3 * mCH3 + nCH4 * mCH4


def write_template(args):

    name = args.get("molecule", "pentane")
    n = args.get("size", 3)
    solid = args.get("solid", "Au")

    # input variables
    density = args.get("density")
    h = args.get("gap_height")  # Angstrom

    nx = 4 * n
    ny = 7 * n
    nz = 3

    slab = FaceCenteredCubic(directions=[[1, 1, -2], [-1, 1, 0], [1, 1, 1]],
                             size=(1, 1, 1), symbol=solid, pbc=(1, 1, 1))

    coords = slab.get_positions()
    mass = slab.get_masses()[0]

    ax, ay, az = np.diag(slab.cell)

    lx = ax * nx
    ly = ay * ny
    lz = az * nz

    volume = lx * ly * h
    molecule_file = f"{name}.lt"
    mFluid = get_mass_alkane(name)
    Nf = round(density * volume / mFluid)
    nxf, nyf, nzf, gap = get_molecule_grid(molecule_file, lx, ly, h, Nf)

    sig_wf = (3.75 + 2.63) / 2.

    with open('moltemplate_files/system.lt', 'w') as f:
        f.write(write_solid_data(coords, mass))
        f.write(write_boundary(lx, ly, 2*lz + gap))
        f.write(write_slab(nx, ny, nz, ax, ay, az, name='solidU', shift=lz+gap))
        f.write(write_slab(nx, ny, nz, ax, ay, az, name='solidL'))
        f.write(write_fluid(molecule_file, Nf, nxf, nyf, nzf, lx, ly, gap, lz+az/6.))
        f.write(write_settings(args, gap - (h + sig_wf/4.)))
        f.write(write_run())


def build_template(args):

    restart_file = args.get("restart_file", "run.in.restart")

    moltemplate_command = ["moltemplate.sh",
                           "-overlay-all",
                           "-lammps-script", "run.in.all",
                           "moltemplate_files/system.lt"
                           ]

    subprocess.run(moltemplate_command, shell=False,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    write_restart(restart_file)


def write_restart(restart_file):
    s = f"""
# ----------------- Load restart file -----------------

read_restart "{restart_file}"

# ----------------- Settings Section -----------------

include "system.in.settings"

# ----------------- Run Section -----------------

include "static/in.flow.lmp"
include "static/in.run.sample.lmp"
"""

    with open("run.in.restart", "w") as f:
        f.write(s)
