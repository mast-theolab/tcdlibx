import numpy as np
from cube_manip import CubeData, VecCubeData
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.conversion_units import edip_conv, mdip_conv

class TenCubData(CubeData):
    """
    CubeData containing Tensor field (3x3 mat)
    """
    def __init__(self, cubedata):
        try:
            if not isinstance(cubedata, CubeData):
                raise NoValidData('TenCubData', 'CubeData object required')
            # +1 is for electric field
            if not int(cubedata.nval / (cubedata.natoms + 1)) == 9:
                raise NoValidData('TenCubData', 'Not tensor field data set. nval = {}'.format(cubedata.nval))
            CubeData.__init__(self, cubedata)
            self.__reshape_cube()
            self.evec = None
        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

    def set_evec(self, evec):
        """
        add evec to the object
        """
        nmn = self.natoms * 3 - 6
        dimens = (nmn, self.natoms * 3)
        if evec.shape == dimens:
            self.evec = evec
        else:
            raise NoValidData('TenCubData', 'Not the same molecule')

    def __reshape_cube(self):
        """
        reshape the cube for the multiplication with evec
        """
        natms = self.natoms
        npts = int(self.npts[0] * self.npts[1] * self.npts[2])
        self.cube = self.cube.reshape(3, 3*(natms + 1), npts)
        self.cube = self.cube[:, 3:, :]

    def explicit_calc(self, nmnum):
        if not self.evec:
            raise NoValidData('explicit_calc', 'Normal modes not set')
        dimens = self.cube.shape[2]
        res = np.zeros((self.cube.shape[0], dimens))
        for i in range(dimens):
            res[:, i] = np.dot(self.cube[:, :, i], self.evec[nmnum - 1])
        return res


    def get_vecvib(self, nmnum):
        """
        selected the normal modes returns the corresponding vector field in
        a VecCubeData object
        """
        if not self.evec:
            raise NoValidData('get_vecvib', 'Normal modes not set')
        nmn = self.natoms * 3 - 6
        if nmnum < 0 or nmnum > nmn:
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
        datares = VecCubeData(datares)
        return datares

class VcdData(VecCubeData):
    """
    Cube data containing Vector file from APT or AAT * L
    """
    def __init__(self, cubedata, typev='ele', ithevec=None, nu_freq=None):
        try:
            if not cubedata.cube.shape[0] == 3:
                raise NoValidData('VtcdData', 'Not vec field data set')
            if typev.lower() not in ['ele', 'mag']:
                raise NoValidData('VtcdData.typev', 'Only ele or mag accepted')
            VecCubeData.__init__(self, cubedata)
            self.evec = ithevec
            self.typev = typev.lower()
            self.energy = nu_freq
            self.mu_tot = None
            self.mg_tot = None

        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

    def _calc_prefactor(self):
        """
        compute the prefactor
        """
        if not self.energy:
            print('Transition energy not set')
            return 1
        if self.typev == 'ele':
            prefactor = edip_conv
            # np.sqrt(cb.PC.au2cm1()/(2*self.energy)) per D
        elif self.typev == 'mag':
            prefactor = mdip_conv
            # per R c'è un due che balla
        else:
            raise NoValidData('VcdData._calc_prefactor', 'Not available')
        return prefactor

    def set_mutot(self, vec):
        """
        TODO
        """
        if not isinstance(vec, np.ndarray):
            raise NoValidData('VcdData.set_mutot', 'ndarray required')
        if vec.shape != (3,):
            raise NoValidData('VcdData.set_mutot', '3 ndarray required')
        self.mu_tot = vec

    def set_mgtot(self, vec):
        """
        TODO
        """
        if not isinstance(vec, np.ndarray):
            raise NoValidData('VcdData.set_mutot', 'ndarray required')
        if vec.shape != (3,):
            raise NoValidData('VcdData.set_mutot', '3 ndarray required')
        self.mg_tot = vec

    def proj_on_vec(self, typ='ele', nucl=False, rot=False, cube=False):
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
        return super(VcdData, self).proj_on_vec(vec=vec, rot=rot, cube=cube)
