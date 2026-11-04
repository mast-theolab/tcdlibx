import numpy as np
from estampes.data.physics import PHYSFACT, PHYSCNST, phys_fact

# Constants
m2ang = 1.0e10
hbar = PHYSCNST.planck*m2ang**2/(2*np.pi*PHYSFACT.amu2kg)
hc = PHYSCNST.planck*PHYSCNST.slight*1.0e18
# -- Conversion from Hartree to eV
AU2EV = PHYSFACT.Eh2J/PHYSFACT.e2C
# -- Normal-mode derivatives, from sqrt(amu).bohr (fchk) to sqrt(me).bohr (au)
DQ_AMU2AU = np.sqrt(phys_fact("au2amu"))
# -- Conversion from mass-weighted to dimensionless normal coords
MWQ2q = 1./(np.sqrt(2*np.pi*PHYSCNST.slight/hbar)*PHYSFACT.bohr2ang)
# -- Conversion from sqrt(Eh/amu) to cm^-1
eval2cm2 = MWQ2q**2/hc * PHYSFACT.Eh2J*1.0e18
# -- Electric dipole from au to statC.cm
edip_conv = phys_fact("au2esu")*PHYSFACT.bohr2ang*1.0e-8
# -- Magnetic dipole from au to statA.cm^2
mdip_conv = 1.0e4*PHYSCNST.planck/(2*np.pi)*phys_fact("au2esu") / PHYSFACT.amu2kg
# -- IR to epsilon: eps = IR2EPS*nu*D*g, D in 1e-40 esu^2.cm^2, nu in cm^-1
IR2EPS = 1.0e-47*8.*np.pi**3*PHYSCNST.avogadro / (3000.*PHYSCNST.planck*PHYSCNST.slight*np.log(10))
# -- VCD to Depsilon: Deps = VCD2DE*nu*R*g, R in 1e-44 esu^2.cm^2
VCD2DE = 1.0e-51*32.*np.pi**3*PHYSCNST.avogadro / (3000.*PHYSCNST.planck*PHYSCNST.slight*np.log(10))
ds_fact = MWQ2q**2 * edip_conv**2*1.0e40/2.
rs_fact = edip_conv*mdip_conv*1.0e44/PHYSCNST.slight


def edip_cgs(vec, freq):
    return vec/np.sqrt(2*freq)*edip_conv*MWQ2q*1e20


def mdip_cgs(vec, freq):
    return vec*mdip_conv/MWQ2q*np.sqrt(2*freq)/PHYSCNST.slight*1e24


def ele_edip_cgs(vec):
    au2Cm = PHYSFACT.e2C*PHYSFACT.bohr2ang*1.0e-10
    return vec * au2Cm * 10 * PHYSCNST.slight


def ele_mdip_cgs(vec):
    au2kg = 1.0e4*PHYSFACT.Eh2J / (PHYSCNST.slight*PHYSCNST.finestruct)**2
    au2JT = PHYSCNST.planck/(2.0*np.pi)*PHYSFACT.e2C/au2kg
    return vec * au2JT * (10**3)


