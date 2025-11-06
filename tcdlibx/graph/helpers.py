import random
import numpy as np
import typing as tp
from estampes.data.atom import atomic_data
# from estampes.tools.atom import convert_labsymb
from estampes.data.physics import PHYSFACT, phys_fact
from tcdlibx.utils.var_tools import getellipsoid
from tcdlibx.calc.cube_manip import AimCubeData, CubeData, VecCubeData, VtcdData
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.conversion_units import edip_cgs, mdip_cgs, ele_mdip_cgs, ele_edip_cgs

def random_colors(ncolors: int) -> list[list[float]]:
    """return a list on n random colors

    Arguments:
        ncolors {int} -- Number of colors
    """
    color = [[random.random() for _ in range(3)]
                 for i in range(ncolors)]
    return color


def filtervecatom(cubefile: CubeData, thresh: float = 0.15) -> list[int]:
    """Returns the indices of voxels too close to the atomic centre
    defined as closer to thresh(default=0.15)*vdw radius in bohr

    Args:
        cubefile (_type_): _description_
        thresh (float, optional): _description_. Defaults to 0.15.
    """
    res = []
    radii = [atomic_data(int(x))[int(x)]['rvdw']/PHYSFACT.bohr2ang*thresh for x in cubefile.ian]
    for i in range(cubefile.crd.shape[0]):
        tmp_index = cubefile.indexinsphere(cubefile.crd[i], radii[i]).tolist()
        if tmp_index:
            res.extend(tmp_index)
    return list(set(res))

def nuclear_tensors(atnums: np.ndarray, atcrds: np.ndarray) -> dict[str, np.ndarray]:
    natoms = atnums.shape[0]
    res = {}
    res['aptnuc'] = np.zeros((natoms*3, 3))
    res['aatnuc'] = np.zeros((natoms*3, 3))
    iden = np.identity(3)
    for i in range(natoms):
        res['aptnuc'][i*3:i*3+3, :] = atnums[i] * iden
        res['aatnuc'][i*3:i*3+3, :] = np.array([[0, -atcrds[i, 2], atcrds[i, 1]],
                                                 [atcrds[i, 2], 0, -atcrds[i, 0]],
                                                 [-atcrds[i, 1], atcrds[i, 0], 0]]).T*atnums[i]/4
    return res

class Molecule():
    """Class to store molecular data organized in a dictionary
    it also stores the fragments and the sampled points for the ellipsoid
    """
    def __init__(self, data):
        self._moldata = data
        self._frags = None
        self._samplepoints = None

    @property
    def natoms(self):
        return len(self._moldata['atnum'])

    @property
    def crd(self) -> np.ndarray:
        return self._moldata['atcrd']

    @property
    def atnum(self) -> np.ndarray:
        return self._moldata['atnum']

    @property
    def atmas(self) -> np.ndarray:
        return self._moldata['atmas']

    @property
    def nfrags(self) -> int:
        if self._frags is None:
            return 0
        else:
            return len(self._frags)

    @property
    def samplepoints(self) -> tp.Union[np.ndarray, None]:
        return self._samplepoints

    def get_com(self, mask=None):
        if mask is None:
            mask = np.ones(self.natoms, dtype=bool)
        return np.average(self.crd[mask, :], axis=0, weights=self.atmas[mask])

    def set_fragment(self, frags: list[list[int]],
     colors: tp.Optional[list[list[float]]] = None):
        # sfrags = [[x-1 for x in item] for item in frags]
        sfrags = frags
        tmp_full = [item for sublist in sfrags for item in sublist]
        natms = len(tmp_full)
        unic_atms = set(tmp_full)
        lunic_atms = len(unic_atms)
        if natms != lunic_atms:
            print("WARNING: some atoms present more than one fragments")
        try:
            unic_atms = np.array(list(unic_atms))
            if (unic_atms < 0).any() or (unic_atms >= self.natoms).any():
                self._frags = None
                print(unic_atms)
                print(sfrags)
                print(frags)
                raise ValueError
            else:
                excluded = list(set(range(self.natoms)) - set(tmp_full))
                if excluded:
                    sfrags.append(excluded)
                self._frags = {}
                self._frags['indx'] = [tuple(x) for x in sfrags]
                if colors and len(colors) == len(sfrags):
                    self._frags['colors'] = colors
                else:
                    self._frags['colors'] = random_colors(len(sfrags))
        except Exception as err:
            print(err)

    def get_frag_colors(self):
        if self._frags is not None:
            return self._frags['colors']
        else:
            raise NoValidData("Molecule.get_frag_colors", "Fragments not defined")

    def get_frag_indx(self):
        if self._frags is not None:
            return self._frags['indx']
        else:
            raise NoValidData("Molecule.get_frag_colors", "Fragments not defined")
    
    def sample_ellipse_space(self, npts: int, scale: float=3,
                             mask: tp.Optional[tp.Union[np.ndarray, None]] = None):
        if mask is None:
            mask = np.ones(self.natoms, dtype=bool)
        self._samplepoints = getellipsoid(self._moldata['atcrd'][mask, :],
                                          np.array(self._moldata['atnum'])[mask],
                                          npts, scale)
            

class VibMolecule(Molecule):
    """ Class to stare data related to vibrational transitions
    """
    def __init__(self, data):
        Molecule.__init__(self, data)
        self._nvib = len(self._moldata['atnum'])*3 - 6
        self._vtcd = {}
        self._aimdata = None
        self._nuctens()

    @property
    def nvib(self):
        return self._nvib

    @property
    def ntrans(self) -> int:
        return self._nvib

    @property
    def apt(self) -> np.ndarray:
        return self._moldata['apt']
    
    @property
    def aat(self) -> np.ndarray:
        return self._moldata['aat']

    def _nuctens(self):
        nuc_tens = nuclear_tensors(np.array(self.atnum),
                                   self.crd)
        self._moldata['aptnuc'] = nuc_tens['aptnuc']
        self._moldata['aptele'] = self._moldata['apt'] - nuc_tens['aptnuc']
        self._moldata['aatnuc'] = nuc_tens['aatnuc']
        self._moldata['aatele'] = self._moldata['aat'] - nuc_tens['aatnuc']

    def get_freq(self, vib: int) -> float:
        if vib > self._nvib - 1  or vib < 0:
            raise NoValidData("Molecule.get_freq", "Vibration out of range")
        return self._moldata['freq'][vib]

    def get_transeng(self, vib: int) -> float:
        if vib > self._nvib - 1 or vib < 0:
            raise NoValidData("Molecule.get_transeng", "Vibration out of range")
        return self._moldata['freq'][vib]/phys_fact("au2cm1")

    def get_dtm(self, vib: int, tps: str = 'tot',
                cgs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if vib > self._nvib - 1 or vib < 0:
            raise NoValidData("Molecule.get_dtm", "Vibration out of range")
        if tps == 'tot':
            res = (self._moldata['edi'][vib],
                   self._moldata['mdi'][vib])
        elif tps == 'ele':
            res = (np.einsum("j,jk->k", self._moldata['lx'][vib], self._moldata['aptele']),
                   np.einsum("j,jk->k", self._moldata['lx'][vib], self._moldata['aatele']))
        elif tps == 'nuc':
            res = (np.einsum("j,jk->k", self._moldata['lx'][vib], self._moldata['aptnuc']),
                   np.einsum("j,jk->k", self._moldata['lx'][vib], self._moldata['aatnuc']))
        else:
            raise NoValidData("Molecule.get_dtm", 'No valid dtm type: tot, ele, nuc')
        if cgs:
            res = (edip_cgs(res[0], self.get_freq(vib)),
                   mdip_cgs(res[1], self.get_freq(vib))) 
        return res      

    def add_tcd(self, state: int, cubedata: VtcdData):
        if state > self._nvib - 1 or state < 0:
            raise NoValidData("Molecule.add_tcd", "Vibration out of range")
        elif cubedata.natoms != self.natoms:
            raise NoValidData("Molecule.add_tcd", "Different molecular systems")
        if self._aimdata is not None:
            self._vtcd[state] = AimCubeData(cubedata, self._aimdata)
            if self._frags is not None:
                self._vtcd[state].set_fragments(self._frags['indx'])
        else:
            self._vtcd[state] = cubedata
    
    def get_tcd_dtm(self, vib: int, tps: str = 'tot',
                     cgs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if vib > self._nvib - 1 or vib < 0:
            raise NoValidData("Molecule.get_vtcd_dtm", "Vibration out of range")
        if tps == "tot":
            res = (self._vtcd[vib].mu_integrate(),
                   self._vtcd[vib].mag_integrate())
        elif tps == "frags":
            # print(self._frags)
            # print(self._aimdata)
            if ((self._frags is None)
              or (self._aimdata is None)):
                raise NoValidData("Molecule.get_tcd_dtm", 'Frags or AIM data not available')
            # FIXME sistemare l'oggetto in cube_manip
            tmp_res = self._vtcd[vib].get_frags_contribution()
            res = (np.array(tmp_res['int']), np.array(tmp_res['rot']))
        else:
            raise NoValidData("Molecule.get_tcd_dtm", 'No valid dtm type: tot, frag')
        if cgs:
            if tps == "frag":
                for i in range(res[0].shape[0]):
                    res = (np.array([edip_cgs(res[0][i, :], self.get_freq(vib))]),
                           np.array([mdip_cgs(res[1][i, :], self.get_freq(vib))]))
            else:
                res = (edip_cgs(res[0], self.get_freq(vib)),
                       mdip_cgs(res[1], self.get_freq(vib))) 
        return res  
           
    def get_evec(self, vib: int) -> np.ndarray:
        if vib > self._nvib - 1 or vib < 0:
            raise Exception("Vibration out of range")
        # Scaled NM
        return self._moldata['evec'][vib].reshape(-1,3)*10 

    def avail_tcd(self) -> list[int]:
        return list(self._vtcd.keys())

    def get_tcd(self, vib: int) -> VtcdData:
        # if vib in self._vtcd.keys():
        return self._vtcd[vib]

    def add_aim(self, aimcube: CubeData):
        # FIXME aim must be added after, and all data are expected to have the same cube
        if not self._vtcd:
            raise NoValidData("add_aim", "Add a VTCD data first")
        elif not self._vtcd[self.avail_tcd()[0]].same_system(aimcube):
            raise NoValidData("AimCube", "must share system with VTCD data")
        # FIXME redundant
        self._aimdata = aimcube
        for tcd in self.avail_tcd():
            self._vtcd[tcd] = AimCubeData(self._vtcd[tcd], self._aimdata)
            if self._frags is not None:
                self._vtcd[tcd].set_fragments(self._frags['indx'])

    def remove_tcd(self, vib: int) -> bool:
        """
        Remove a specific TCD cube from memory to free resources.
        
        Args:
            vib (int): Vibration index to remove
            
        Returns:
            bool: True if cube was removed, False if not found
        """
        if vib in self._vtcd:
            del self._vtcd[vib]
            return True
        return False

    def remove_all_tcd(self) -> int:
        """
        Remove all stored TCD cubes from memory to free resources.
        
        Returns:
            int: Number of cubes removed
        """
        count = len(self._vtcd)
        self._vtcd.clear()
        return count


class EleMolecule(Molecule):
    """ Class to store data related to electronic transitions """
    def __init__(self, data):
        Molecule.__init__(self, data)
        self._nstates = self._moldata['exeng'].shape[0]
        self._etcd = {}
        self._aimdata = None

    @property
    def nstates(self) -> int:
        return self._nstates

    @property
    def ntrans(self) -> int:
        return self._nstates

    def get_exeng(self, state: int) -> float:
        if state > self._nstates-1 or state < 0:
            raise Exception("State out of range")
        return self._moldata['exeng'][state]

    def get_transeng(self, state: int) -> float:
        if state > self._nstates-1 or state < 0:
            raise Exception("State out of range")
        return self._moldata['exeng'][state]

    def get_edtm(self, state: int) -> float:
        if state > self._nstates-1 or state < 0:
            raise Exception("State out of range")
        return -1*self._moldata['edi'][state]/self._moldata['exeng'][state]

    def get_mdtm(self, state: int) -> float:
        if state > self._nstates-1 or state < 0:
            raise Exception("State out of range")
        return self._moldata['mdi'][state] * -.5

    def get_dtm(self, state: int, tps: str="ele", cgs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            state (int): _description_
            tps (str, optional): Note used at the moment just for compatibility. Defaults to "ele".
            cgs (bool, optional): _description_. Defaults to True.

        Raises:
            NoValidData: _description_

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        if state > self._nstates - 1  or state < 0:
            raise NoValidData("EleMolecule.get_dtm", "State out of range")
        res = (-1*self._moldata['edi'][state] / self._moldata['exeng'][state],
               self._moldata['mdi'][state] * -.5)
        if cgs:
            res = (ele_edip_cgs(res[0]),
                   ele_mdip_cgs(res[1]))
        # print(res)
        return res      

    def add_tcd(self, state: int, cubedata: VecCubeData):
        if state > self._nstates - 1 or state < 0:
            raise NoValidData("EleMolecule.add_tcd", "State out of range")
        elif cubedata.natoms != self.natoms:
            raise NoValidData("EleMolecule.add_tcd", "Different molecular systems")
        if self._aimdata is not None:
            self._etcd[state] = AimCubeData(cubedata, self._aimdata)
            if self._frags is not None:
                self._etcd[state].set_fragments(self._frags['indx'])
        else:
            self._etcd[state] = cubedata
    
    def get_tcd_dtm(self, state: int, tps: str = 'tot',
                     cgs: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if state > self._nstates -1  or state < 0:
            raise NoValidData("EleMolecule.get_vtcd_dtm", "State out of range")
        if tps == "tot":
            res = (self._etcd[state].integrate() / self._moldata['exeng'][state], # to length
                   self._etcd[state].rotorintegrate()/2)
        elif tps == "frags":
            # print(self._frags)
            # print(self._aimdata)
            if ((self._frags is None)
              or (self._aimdata is None)):
                raise NoValidData("Molecule.get_tcd_dtm", 'Frags or AIM data not available')
            # FIXME sistemare l'oggetto in cube_manip
            tmp_res = self._etcd[state].get_frags_contribution()
            res = (np.array(tmp_res['int']), np.array(tmp_res['rot']))
        else:
            raise NoValidData("EleMolecule.get_tcd_dtm", 'No valid dtm type: tot, frag')
        if cgs:
            if tps == "frag":
                for i in range(res[0].shape[0]):
                    res = (np.array([ele_edip_cgs(res[0][i, :])]),
                           np.array([ele_mdip_cgs(res[1][i, :])]))
            else:
                res = (ele_edip_cgs(res[0]),
                       ele_mdip_cgs(res[1])) 
        # print(res)
        return res  
           
    def avail_tcd(self) -> list[int]:
        return list(self._etcd.keys())

    def get_tcd(self, state: int) -> VecCubeData:
        # if vib in self._vtcd.keys():
        return self._etcd[state]

    def add_aim(self, aimcube: CubeData):
        # FIXME aim must be added after, and all data are expected to have the same cube
        if not self._etcd:
            raise NoValidData("add_aim", "Add a VTCD data first")
        elif not self._etcd[self.avail_tcd()[0]].same_system(aimcube):
            raise NoValidData("AimCube", "must share system with VTCD data")
        # FIXME redundant
        self._aimdata = aimcube
        for tcd in self.avail_tcd():
            self._etcd[tcd] = AimCubeData(self._etcd[tcd], self._aimdata)
            if self._frags is not None:
                self._etcd[tcd].set_fragments(self._frags['indx'])

    def remove_tcd(self, state: int) -> bool:
        """
        Remove a specific TCD cube from memory to free resources.
        
        Args:
            state (int): Electronic state index to remove
            
        Returns:
            bool: True if cube was removed, False if not found
        """
        if state in self._etcd:
            del self._etcd[state]
            return True
        return False

    def remove_all_tcd(self) -> int:
        """
        Remove all stored TCD cubes from memory to free resources.
        
        Returns:
            int: Number of cubes removed
        """
        count = len(self._etcd)
        self._etcd.clear()
        return count
    

class MyvtkActor():
    """Helper class to store vtk actor and filter and allow an easy access to them
    """

    def __init__(self, actor, filter=None):
        self._actor = actor
        self._filter = filter

    @property
    def actor(self):
        return self._actor

    @property
    def filter(self):
        return self._filter

    def GetCenter(self):
        """Get the center of the actor"""
        return self._actor.GetCenter()

def fibonacci_spiral_samples_on_unit_sphere(nb_samples, mode=0):
    """
    Generate points on a unit sphere using Fibonacci spiral sampling.
    Taken from:
    https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    
    Args:
        nb_samples (int): Number of sample points on the sphere
        mode (int): Sampling mode (0 for deterministic, other for random shift)
        
    Returns:
        np.ndarray: Array of shape (nb_samples, 3) with unit sphere coordinates
    """
    shift = 1.0 if mode == 0 else nb_samples * np.random.random()
 
    ga = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / nb_samples
    
    ss = np.zeros((nb_samples, 3))
    j = 0
    for i in range(nb_samples):
        phi = ga * ((i + shift) % nb_samples)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = ((i + 0.5) * offset) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        ss[j, :] = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
        j += 1
    return ss

DEFAULT_VIS_VALS = {'isoval': {'iso': 0.01},
                         'vfield': {'vfmax': 1e2,
                                    'vfmin': 1e5,
                                    'mspeed': None,
                                    'npoints': 100,
                                    'scalellipse': 3.,
                                    'showdir': False,
                                    'showell': False,
                                    'conescale': .1,
                                    'showbar': False,
                                    'animate_particles': False,
                                    'num_particles': 15,
                                    'particle_type': 'sphere'},
                         'quiver': {'scale': 100,
                                   'subsample': 5},
                         'tensor': {'sphere_radius': 0.5,
                                    'nb_sphere_samples': 50,
                                    'nb_sphere_surface': 250,
                                    'vector_scale': 1.0,
                                    'surface_scale': 1.0,
                                    'opacity': 0.8,
                                    'color_scheme': 'weight_mag',
                                    'surf_color_scheme': 'magnitude',
                                    'atom_filter': None,
                                    'show_spheres': False}}