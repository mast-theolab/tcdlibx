import numpy as np
from tcdlibx.calc.cube_manip import CubeData, VecCubeData
from tcdlibx.utils.custom_except import NoValidData
from estampes.data.physics import PHYSFACT, PHYSCNST, phys_fact

class TenCubData(CubeData):
    """
    CubeData containing Tensor field (3x3 mat)
    """
    def __init__(self, cubedata: CubeData) -> None:
        if not isinstance(cubedata, CubeData):
            raise NoValidData('TenCubData', 'CubeData object required')
        # +1 is for electric field
        if not int(cubedata.nval / (cubedata.natoms + 1)) == 9:
            raise NoValidData('TenCubData', 'Not tensor field data set. nval = {}'.format(cubedata.nval))
        CubeData.__init__(self, cubedata)
        self.__reshape_cube()
        self.evec: np.ndarray | None = None

    def set_evec(self, evec: np.ndarray) -> None:
        """
        add evec to the object
        """
        nmn = self.natoms * 3 - 6
        dimens = (nmn, self.natoms * 3)
        if evec.shape == dimens:
            self.evec = evec
        else:
            raise NoValidData('TenCubData', 'Not the same molecule')

    def __reshape_cube(self) -> None:
        """
        reshape the cube for the multiplication with evec
        """
        natms = self.natoms
        npts = int(self.npts[0] * self.npts[1] * self.npts[2])
        self.cube = self.cube.reshape(3, 3*(natms + 1), npts)
        self.cube = self.cube[:, 3:, :]

    def explicit_calc(self, nmnum: int) -> np.ndarray:
        if self.evec is None:
            raise NoValidData('explicit_calc', 'Normal modes not set')
        dimens = self.cube.shape[2]
        res = np.zeros((self.cube.shape[0], dimens))
        for i in range(dimens):
            res[:, i] = np.dot(self.cube[:, :, i], self.evec[nmnum - 1])
        return res


    def get_vecvib(self, nmnum: int) -> VecCubeData:
        """
        selected the normal modes returns the corresponding vector field in
        a VecCubeData object
        """
        if self.evec is None:
            raise NoValidData('get_vecvib', 'Normal modes not set')
        nmn = self.natoms * 3 - 6
        if nmnum < 1 or nmnum > nmn:
            raise NoValidData('get_vecvib', 'Normal modes out of range 1 - {}'.format(nmn))
        #tmpc = self.explicit_calc(nmnum)
        tmpc = np.einsum('ijk,ij->ik', self.cube, self.evec[np.newaxis, nmnum - 1])
        datares = CubeData()
        datares.natoms = self.natoms
        datares.npts = self.npts
        datares.loc2wrd = self.loc2wrd
        datares.ian = self.ian
        datares.crd = self.crd
        datares.nval = 3
        datares.cube = tmpc
        return VecCubeData(datares)

class VcdData(VecCubeData):
    """
    Cube data containing Vector file from APT or AAT * L
    """
    def __init__(self, cubedata: CubeData, typev: str = 'ele',
                 ithevec: np.ndarray | None = None,
                 nu_freq: float | None = None) -> None:
        if not cubedata.cube.shape[0] == 3:
            raise NoValidData('VcdData', 'Not vec field data set')
        if typev.lower() not in ['ele', 'mag']:
            raise NoValidData('VcdData.typev', 'Only ele or mag accepted')
        VecCubeData.__init__(self, cubedata)
        self.evec = ithevec
        self.typev = typev.lower()
        self.energy = nu_freq
        self.mu_tot: np.ndarray | None = None
        self.mg_tot: np.ndarray | None = None

    def _calc_prefactor(self) -> float:
        """
        compute the prefactor
        """
        if not self.energy:
            print('Transition energy not set')
            return 1
        if self.typev == 'ele':
            prefactor = phys_fact("au2esu")*PHYSFACT.bohr2ang*1.0e-8
            # np.sqrt(cb.PC.au2cm1()/(2*self.energy)) per D
        elif self.typev == 'mag':
            prefactor = 1.0e4*PHYSCNST.planck/(2*np.pi)*phys_fact("au2esu") / PHYSFACT.amu2kg
            # per R c'è un due che balla
        else:
            raise NoValidData('VcdData._calc_prefactor', 'Not available')
        return prefactor

    def set_mutot(self, vec: np.ndarray) -> None:
        """
        TODO
        """
        if not isinstance(vec, np.ndarray):
            raise NoValidData('VcdData.set_mutot', 'ndarray required')
        if vec.shape != (3,):
            raise NoValidData('VcdData.set_mutot', '3 ndarray required')
        self.mu_tot = vec

    def set_mgtot(self, vec: np.ndarray) -> None:
        """
        TODO
        """
        if not isinstance(vec, np.ndarray):
            raise NoValidData('VcdData.set_mgtot', 'ndarray required')
        if vec.shape != (3,):
            raise NoValidData('VcdData.set_mgtot', '3 ndarray required')
        self.mg_tot = vec

    def proj_on_vec(self, typ: str = 'ele', nucl: bool = False,  # pyright: ignore[reportIncompatibleMethodOverride]
                    rot: bool = False, cube: bool = False) -> np.ndarray | CubeData:
        """
        compute the scalar field, projecting the vector field
        on the electric dipole transition moment
        params nucl: include the nuclear contribution
        """
        if not nucl and not typ == self.typev:
            raise NoValidData('VcdData.proj_on_vec', 'not available')
        if nucl and self.typev == 'ele':
            vec = self.mu_tot
        elif nucl and self.typev == 'mag':
            vec = self.mg_tot
        else:
            vec = self.integrate() # * self._calc_prefactor()
        if vec is None:
            raise NoValidData('VcdData.proj_on_vec', 'total dipole not set (set_mutot/set_mgtot)')
        return super(VcdData, self).proj_on_vec(vec=vec, rot=rot, cube=cube)
