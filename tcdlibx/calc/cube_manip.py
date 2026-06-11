#!/bin/python3
"""
Library to manage the cube file
"""
import os
import importlib.util
import copy
from math import ceil
import numpy as np
# custom libraries
from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.var_tools import angle_between, Poliedron, trilinerarinterpolation
import tcdlibx.utils.types as mytp
GPU = importlib.util.find_spec('numba')
if GPU:
    from tcdlibx.calc.numba_linalg_3 import cross3
    GPU = True
    print('GPU = {}'.format(GPU))


def modl(vec: np.ndarray) -> float:
    """
    return the module of a vector
    """
    return np.sqrt(np.dot(vec, vec))


def mynorm(vec: np.ndarray) -> np.ndarray:
    """
    return the normalized vector
    """
    return vec/modl(vec)


def mu_integrate(cubdata: 'CubeData', mask: np.ndarray | None = None) -> np.ndarray:
    """integrate vector field in all space"""
    if mask is not None:
        mask.any()
        vec_1 = cubdata.cube[:, mask]
    else:
        vec_1 = cubdata.cube
    poligon = cubdata.get_voxvol()
    integrated = vec_1.sum(axis=1) * poligon * -1 * np.sqrt(2)  # Vedi franco
    return integrated


def calc_rot(cubdata: 'CubeData', mask: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate the rotor of a a vector fields
    """
    if not cubdata.box.size:
        cubdata.make_box()
    if mask is not None:
        vec_1 = cubdata.cube[:, mask]
        box_1 = cubdata.box[:, mask]
    else:
        vec_1 = cubdata.cube
        box_1 = cubdata.box
    if GPU:
        res = np.zeros_like(vec_1)
        for i_th in range(vec_1.shape[1]):
            cross3(box_1[:, i_th], vec_1[:, i_th], res[:, i_th])
    else:
        res = np.cross(box_1, vec_1, axisa=0, axisb=0).T
    return res


def mag_integrate(cubdata: 'CubeData', mask: np.ndarray | None = None) -> np.ndarray:
    """integrate vector field in all space"""
    poligon = cubdata.get_voxvol()
    res = calc_rot(cubdata, mask)
    integrated = res.sum(axis=1) * poligon * -1 * np.sqrt(2) # hbar=1
    # to get gaussian value(
    return integrated  # to the magnetic dipole moment


def mask_cube(datacub: 'CubeData', mu_tot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask the vector in base at the rotation"""
    vec_copy = np.copy(datacub.cube)
    vec_1 = np.copy(datacub.cube)
    for i_th in range(vec_copy.shape[1]):
        tmp_mag = np.cross(datacub.box[:, i_th], datacub.cube[:, i_th])
        angle = angle_between(tmp_mag, mu_tot)
        if angle < 1.39:
            vec_copy[:, i_th] = 0.
        elif angle > 1.74:
            vec_1[:, i_th] = 0.
        else:
            vec_1[:, i_th] = 0.
            vec_copy[:, i_th] = 0.
    return vec_1, vec_copy


class CubeData:
    """
    Object to manage the discrete volumetric data set from a gaussian cube file
    """
    def __init__(self, origin: 'CubeData | None' = None) -> None:
        if origin is None:
            self.npts = [0, 0, 0]
            self.loc2wrd = np.identity(4)
            self.wrd2loc = np.identity(4)
            self.natoms = 0
            self.ian = np.array([])
            self.crd = np.array([])
            self.nval = 0
            self.cube = np.array([])
            self.box = np.array([])
        elif isinstance(origin, CubeData):
            self.__copy_constructor(origin)
        else:
            raise NoValidData('CubeData',
                              'origin must be an other CubeData Object')

    def __copy_constructor(self, origin: 'CubeData') -> None:
        """
        Copy Constructor # TODO rewrite in pythonic way
        """
        self.npts = copy.deepcopy(origin.npts)
        self.loc2wrd = copy.deepcopy(origin.loc2wrd)
        self.wrd2loc = copy.deepcopy(origin.wrd2loc)
        self.natoms = origin.natoms
        self.ian = copy.deepcopy(origin.ian)
        self.crd = copy.deepcopy(origin.crd)
        self.nval = origin.nval
        self.cube = copy.deepcopy(origin.cube)
        self.box = copy.deepcopy(origin.box)

    def set_origin(self, origin: mytp.Array3F) -> None:
        """
        Set the origin of the cube
        """
        self.loc2wrd[:3, 3] = np.array(origin)
        self.wrd2loc = np.linalg.inv(self.loc2wrd)

    def get_origin(self) -> mytp.Array3F:
        """
        Get the origin of the cube in world space
        """
        return self.loc2wrd[:3, 3]

    def set_natoms(self, natoms: int) -> None:
        """
        Set the atom present in the cube
        """
        self.natoms = natoms
        self.crd = np.zeros((natoms, 3))
        self.ian = np.zeros(natoms)

    # def set_axstep(self, ithvec, axis):
    #     """
    #     set the step along the specified axis
    #     """
    #     if axis == 'x':
    #         axis = 0
    #     elif axis == 'y':
    #         axis = 1
    #     elif axis == 'z':
    #         axis = 2
    #     self.loc2wrd[:3, axis] = ithvec

    def get_axstep(self, axis: int | str) -> mytp.Array3F:
        """
        return the step along the specified axis
        """
        try:
            axis = int(axis)
            if axis > 2:
                raise NoValidData('get_axstep', '{} not valid value: 0, 1, 2 or x, y, z'.format(axis))
        except ValueError:
            if axis == 'x':
                axis = 0
            elif axis == 'y':
                axis = 1
            elif axis == 'z':
                axis = 2
            else:
                raise NoValidData('get_axstep', '{} not valid value: 0, 1, 2 or x, y, z'.format(axis))
        return self.loc2wrd[:3, axis]

    def get_axstep_norm(self, axis: int | str) -> float:
        """
        return the step along the specified axis
        """
        vec = self.get_axstep(axis)
        return np.sqrt(np.dot(vec, vec))

    def set_loc2wrd(self, x_th: mytp.Array3F,
                    y_th: mytp.Array3F,
                    z_th: mytp.Array3F,
                    ori: mytp.Array3F) -> None:
        """
        Set the local2word 4x4 matrix from np.vector
        x axis, y axis, z axis, origin
        """
        self.loc2wrd[:3, 0] = x_th
        self.loc2wrd[:3, 1] = y_th
        self.loc2wrd[:3, 2] = z_th
        self.loc2wrd[:3, 3] = ori
        self.wrd2loc = np.linalg.inv(self.loc2wrd)

    def _wrdtolocal(self, point: mytp.Array3F) -> mytp.Array3F:
        """
        converts a point from world to local frame
        """
        return np.einsum('ij,j->i', self.wrd2loc,
                         np.append(point, 1))[:3]

    def _getcubeinfo(self, point: np.ndarray) -> tuple[dict[str, list], np.ndarray]:
        """
        * @brief	Helper method that, given the LOCAL coordinates of a point
        *			within the grid (localCoords), returns a Cube structure
        *			(outCube) containing the the LOCAL coordinates of the eight
        *			vertices of the cube (bounding the specified point) and
        *			their associated values. Furthermore, this method also
        *			returns the noormalized coordinates (in the range [0,1])
        *			of the specified point within the cube (outNormCoords).
        *
        """
        cubecorner_relativepositions = [[0, 0, 0],
                                        [0, 0, 1],
                                        [0, 1, 0],
                                        [0, 1, 1],
                                        [1, 0, 0],
                                        [1, 0, 1],
                                        [1, 1, 0],
                                        [1, 1, 1]]
        cubeinfo = {'vox_c': [],
                    'vox_v': []}
        out_normcoords = np.zeros(3)
        cube_origin = (np.abs(point)).astype(int)
        for i in range(3):
            if cube_origin[i] == (self.npts[i]-1):
                cube_origin[i] -= 1
                out_normcoords[i] = 1.0
            else:
                out_normcoords[i] = point[i] - cube_origin[i]

        cube_dim = self.nval
        for relpos in cubecorner_relativepositions:
            voxel_coord = (cube_origin + np.array(relpos)).astype(int)
            # print("voxcoor: {}".format(voxel_coord.tolist()))
            cubeinfo['vox_c'].append(voxel_coord)
            # Since up to now are linear stored
            # TODO move to 3D storage
            # cubeinfo['vox_v'].append(self.cube[tuple(voxel_coord)])
            index = (self.npts[1] * voxel_coord[0] + voxel_coord[1]) * self.npts[2] + voxel_coord[2]
            # print("index: {}".format(index))
            # print("{a[0]}*{b[1]}*{b[2]}+{a[1]}*{b[2]}+{a[2]}".format(a=voxel_coord.tolist(), b= self.npts))
            if  cube_dim == 1:
                cubeinfo['vox_v'].append(self.cube[index])
            # elif cube_dim == 2:
            else:
                cubeinfo['vox_v'].append(self.cube[:, index])
            # else:
            #    cubeinfo['vox_v'].append(self.cube[tuple(voxel_coord)])

        return (cubeinfo, out_normcoords)

    def _cubetolinear(self, voxel: np.ndarray) -> int:
        return (self.npts[1] * voxel[0] + voxel[1]) * self.npts[2] + voxel[2]

    def _lineartocube(self, point: int) -> list[int]:
        return [int(point/(self.npts[2] * self.npts[1])),
                int(point/self.npts[2]) % self.npts[1],
                point % self.npts[2]]

    def get_value(self, point: np.ndarray) -> np.ndarray | float:
        """
        return the value at the provided point if lies inside
        the grid
        """
        loc_point = self._wrdtolocal(point)
        if (loc_point < 0.).any() or (loc_point - np.array(self.npts) > 0).any():
            raise NoValidData('get_value', 'The point lies outside the grid')
        loc_point = np.round(loc_point, 5)
        # returns the point on the grid if it is on it
        if np.isclose(loc_point, np.round(loc_point), atol=1e-5).all():
            return self.cube[loc_point.astype(int)]
        datavox, normpoint = self._getcubeinfo(loc_point)
        return trilinerarinterpolation(datavox['vox_v'], normpoint)

    def _get_value(self, point: np.ndarray) -> np.ndarray | float:
        """
        no checks
        """
        # loc_point = self._wrdtolocal(point)
        datavox, normpoint = self._getcubeinfo(point)
        return trilinerarinterpolation(datavox['vox_v'], normpoint)

    def set_cube(self, cube_da: np.ndarray) -> None:
        """
        Set the cube dataset, consistency checked with npts and nval
        """
        # BUG accept also list? decide and fix
        try:
            dimen = cube_da.shape
            nptss = self.npts[0] * self.npts[1] * self.npts[2]
            if not dimen[0] == self.nval or not dimen[1] == nptss:
                raise NoValidData('set_cube', 'ndarray {}x{} expected, {}x{} provided'.format(self.nval, nptss, dimen[0], dimen[1]))
        except AttributeError:
            raise NoValidData('CubeData.set_cube', 'expected np.ndarray')

    def add_atom(self, i_an: int, coord: np.ndarray, index: int) -> None:
        """
        add an atom number and its coordinate
        """
        self.ian[index] = i_an
        self.crd[index, :] = coord

    def make_box(self) -> None:
        """
        build the vectors of coordinate in world space
        """
        # Define the local Grid
        cubol = np.mgrid[0:self.npts[0]:1,
                         0:self.npts[1]:1,
                         0:self.npts[2]:1].reshape(3, -1)
        # Multiply to local2World to get the world Grid
        # np.inserd add the 1 before the rototraslation
        self.box = np.dot(self.loc2wrd, np.insert(cubol, 3, 1, axis=0))[:3, :]

    def get_voxvol(self) -> float:
        """
        return the volume of the voxel
        """
        poligon = np.dot(self.loc2wrd[:3, 0],
                         np.cross(self.loc2wrd[:3, 1],
                                  self.loc2wrd[:3, 2]))
        return poligon

    def same_system(self, other: 'CubeData') -> bool:
        """
        Check if two cube are related to the same system and if they are
        the same type of data
        """
        res = False
        if (self.npts == other.npts and
                (self.loc2wrd == other.loc2wrd).all() and
                (self.ian == other.ian).all() and
                ((self.crd-other.crd) < 2.e-6).all()):  # and
                # self.nval == other.nval):
            res = True

        return res

    def __add__(self, other: 'CubeData') -> 'CubeData | None':
        """
        add two volumetric data set if defined in the same world space
        """
        try:
            if not self.same_system(other):
                raise NoValidData("CubeSum", "No same systems in the two cube")
            tmp = copy.deepcopy(self)
            tmp.cube = self.cube + other.cube
            return tmp
        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

#    def __iadd__(self, other):
#        """
#        overload +=
#        """
#        self = self + other
#        return self

    def __sub__(self, other: 'CubeData') -> 'CubeData':
        """
        subtract two volumetric data set if defined in the same world space
        """
        # try:
        #     if not self.same_system(other):
        #         raise NoValidData("CubeSum", "No same systems in the two cube")
        tmp = copy.deepcopy(self)
        tmp.cube = self.cube - other.cube
        return tmp
        # except NoValidData as err:
        #     print("{}:{}".format(err.expression, err.message))

    def __mul__(self, param: float) -> 'CubeData':
        """
        multiply the data set per parameters
        """
        tmp = copy.deepcopy(self)
        tmp.cube = self.cube * param

        return tmp

#    def __imul__(self, param):
#        """
#        overload *=
#        """
#        self = self * param
#        return self

    def __truediv__(self, param: float) -> 'CubeData':
        """
        overload true division
        """
        tmp = copy.deepcopy(self)
        tmp.cube = self.cube / param

        return tmp
    # BUG
    # def rotate(self, vec_tr):
    #     """
    #     Rotates the cube data set such as the z
    #     axis match the provided vector
    #     """
    #     z_ax = np.array([0., 0., 1.])
    #     rot = rotation_matrix(vec_tr, z_ax)
    #     i123 = 0
    #     for _ in range(0, self.npts[0]):
    #         for _ in range(0, self.npts[1]):
    #             for _ in range(0, self.npts[2]):
    #                 self.cube[:, i123] = np.dot(rot, self.cube[:, i123])
    #                 self.box[:, i123] = np.dot(rot, self.box[:, i123])
    #                 i123 += 1
#   #      self.cube = np.dot(np.transpose(rot), self.cube)
#   #      self.box = np.dot(np.transpose(rot), self.box)
    #     return rot

    # def closergridpoint(self, point):

    def indexinsphere(self, center: np.ndarray, radius: float) -> np.ndarray:
        """
        return the indices of grid points in a sphere centered at center and with radius = radius
        only points within the box are considered
        """
        cubrepos = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 1, 1],
                             [1, 0, 0],
                             [1, 0, 1],
                             [1, 1, 0],
                             [1, 1, 1]])
        loc_point = self._wrdtolocal(center)
        if (loc_point < 0.).any() or (loc_point - np.array(self.npts) > 0).any():
            raise NoValidData('get_value', 'The point lies outside the grid')
        loc_point = np.round(loc_point, 5)
        # BUG assumed an orthonormal set of coordinate
        # select a rectangular box and then refine
        # diag(wrd2loc) = 1/step_size, so multiply to get radius in grid-step units
        rad_local = np.ceil(np.repeat(radius, 3) * np.abs(np.diag(self.wrd2loc)[:3]))
        # Find the closest voxel
        cube_origin = np.floor(loc_point)
        deltas = cubrepos - (cube_origin[np.newaxis, :] - cube_origin)
        normval = (deltas**2).sum(axis=1)
        clos_voxel = cubrepos[normval.argmin()]+cube_origin
        # Building the box
        minpos = clos_voxel - rad_local
        nsteps = rad_local*2+1
        for i in range(3):
            if minpos[i] < 0:
                nsteps[i] += minpos[i]
                minpos[i] = 0
            if minpos[i] + nsteps[i] > self.npts[i]:
                nsteps[i] = self.npts[i] - minpos[i]
        # print(minpos, nsteps)
        box = np.mgrid[0:nsteps[0]:1,
                       0:nsteps[1]:1,
                       0:nsteps[2]:1].reshape(3, -1)+minpos[:, np.newaxis]
        # Multiply to local2World to get the world Grid
        # np.inserd add the 1 before the rototraslation
        cube_world = np.dot(self.loc2wrd, np.insert(box, 3, 1, axis=0))[:3, :]
        # in the sphere
        deltas = cube_world-center[:, np.newaxis]
        insphere = np.sqrt(np.einsum("ij,ij->j", deltas, deltas)) < radius
        indices = np.array(box[2, insphere] + self.npts[2] *(box[1, insphere]+ self.npts[1] * box[0, insphere]), dtype=int)
        return indices

    def integrate(self, mask: np.ndarray | None = None) -> float:
        """integrate vector field in all space
        param mask: boolen mask
        # TODO check the dimensions
        """
        if mask is not None:
            mask.any()
            vec_1 = self.cube[mask]
        else:
            vec_1 = self.cube
        poligon = self.get_voxvol()
        integrated = vec_1.sum()*poligon
        return integrated


class ScalarsCube(CubeData):
    """
    CubeData containing A collection of scalar fields
    """
    def __init__(self, cubedata: CubeData) -> None:
        try:
            if not isinstance(cubedata, CubeData):
                raise NoValidData('ScalarsCubData', 'CubeData object required')
            CubeData.__init__(self, cubedata)
            self.labels = [str(x) for x in range(self.nval)]
            self.centroids = None
        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

    def set_labels(self, labels: list[str]) -> None:
        """
        set the labels
        """
        # BUG ?
        if len(labels) == self.nval:
            self.labels = labels
        else:
            print("length not matching the number of scalar fields")

    def _calc_centroids(self) -> None:
        """
        TODO
        """
        # Probabile BUG
        cbsq = self.cube**2
        self.make_box()
        weigh = cbsq.sum(axis=1)
        res = np.einsum('ij,kj->ik', cbsq, self.box)
        self.centroids = res/weigh[:, np.newaxis]

    def get_centroids(self) -> np.ndarray | None:
        """
        TODO
        """
        if not self.centroids:
            self._calc_centroids()
        return self.centroids

    def integrate_orthoplane(self, origin: np.ndarray | list, orth_axis: np.ndarray | list) -> list[list]:
        """
        integrate the planes orthogonal to the provided axis
        """
        origin = np.array(origin)
        orth_axis = np.array(orth_axis)
        norm_axis = mynorm(orth_axis)
        ax_def = np.where(np.abs(1 - norm_axis) < 1e-5)
        if ax_def[0].size:
            it0 = int(ax_def[0][0])
            res = self._integrate_plane_fix(it0)
            if (np.abs(origin[it0]) > 1e-5):
                for item in res[0]:
                    item -= origin[it0]
            if orth_axis[it0] < 0:
                res = [list(reversed(res[0])),
                       list(reversed(res[1]))]
            return res
        return self._integrate_custm_pln(origin, orth_axis)

        # sistemare la selezione del

    def _integrate_plane_fix(self, orth_axis: int) -> list[list]:
        res = [[], []]
        it0 = int(orth_axis)
        i_one = 1 >> it0
        i_two = 2 - (1 >> (2 - it0))

        # Computes deltaZ and minZ
        tmp_res = self.cube.reshape(self.npts).sum(axis=i_two).sum(i_one)

        # Computes the area of the parallelogram having dx and dy as sides
        parrallelogram_area = modl(np.cross(self.get_axstep(i_one),self.get_axstep(i_two)))

        # Computes the axis values
        delta = modl(self.get_axstep(it0))
        res[0] = [x * delta + self.get_origin()[it0] for x in range(self.npts[it0])]

        res[1] = (tmp_res * parrallelogram_area).tolist()

        return res

    def _integrate_custm_pln(self, start_p: np.ndarray, orth_direct: np.ndarray) -> list[list]:
        """
         @brief	Integrates, for each value of t along the specified axis,
        			the function sampled into the grid along the plane
        			orthogonal to the axis.

         @note	The computation performed by this method is accurate only
        			if the function represented by the grid is ~= 0 near the
        			edges and outside the grid!

         @param start_p	The starting point for the integration.
        						Must lies inside the grid!

         @param orth_direct	The direction along which the integration planes
        					are determined. Must be a versor! (lenght = 1)


         @throw NoValidData	If "origin" lies outside the grid.
        """
        origin = self.get_origin()
        bbox = Poliedron(origin,
                         self.get_axstep(0) * (self.npts[0]-1),
                         self.get_axstep(1) * (self.npts[1]-1),
                         self.get_axstep(2) * (self.npts[2]-1))
        z_prime = mynorm(orth_direct)
        if not bbox.isinside(start_p):
            raise NoValidData('integrate_ax',
                              'The specified starting point is out of the bounds of the grid')
        # Searches the local axis which is "almost orthogonal" to z_prime.
        # It will be a sort of x_prime.
        z_min_indx = np.argmin(np.abs(z_prime))

        alm_orth_zprime = np.zeros(3)
        alm_orth_zprime[z_min_indx] = 1
        # X' and Y' are rispectively the X and Y axis of the
        # custom reference frame used for the integration
        y_prime = np.cross(z_prime, alm_orth_zprime)
        y_prime = mynorm(y_prime)
        x_prime = np.cross(y_prime, z_prime)
        x_prime = mynorm(x_prime)
        # Delta step for ray-marchin in local frame
        delta_loc = 0.5
        # Compute the delta along the x', y', z' in world ref
        delta_wrd = delta_loc * min(min(self.get_axstep_norm(2),
                                        self.get_axstep_norm(1)),
                                    self.get_axstep_norm(0))
        # We need to find a range along the Z' direction on which to perform
        # the integration. To do so, we projects the 8 vertices of the bounding
        #  parallelepiped of the grid on the ray: P_Z' = O' + (t * Z'). Then we choose
        # the min and max values of t among those of the 8 vertices of the box.

        verticesbox = []
        verticesbox.append(origin)
        verticesbox.append(origin + (self.npts[2]-1) * self.get_axstep(2))
        verticesbox.append(origin + (self.npts[1]-1) * self.get_axstep(1))
        verticesbox.append(origin + (self.npts[1]-1) * self.get_axstep(1)
                           + (self.npts[2]-1) * self.get_axstep(2))
        verticesbox.append(origin + (self.npts[0]-1) * self.get_axstep(0))
        verticesbox.append(origin + (self.npts[0]-1) * self.get_axstep(0)
                           + (self.npts[2]-1) * self.get_axstep(2))
        verticesbox.append(origin + (self.npts[0]-1) * self.get_axstep(0)
                           + (self.npts[1]-1) * self.get_axstep(1))
        verticesbox.append(origin + (self.npts[0]-1) * self.get_axstep(0)
                           + (self.npts[2]-1) * self.get_axstep(2)
                           + (self.npts[1]-1) * self.get_axstep(1))

        verticesbox = np.array(verticesbox)
        t_tmp = np.einsum('j,ij->i', z_prime, verticesbox)
        t_zprime_min = t_tmp.min()
        t_zprime_max = t_tmp.max()
        del t_tmp

        # Express t_zprime_min and t_zprime_max as multiples of delta.
        # The result is rounded up for i_zprime_min and rounded down for
        # i_zprime_max, so to be sure to remain inside the grid.
        i_zprime_min = int(np.ceil(t_zprime_min / delta_wrd))
        i_zprime_max = int(np.floor(t_zprime_max / delta_wrd))
        if i_zprime_min > i_zprime_max:
            i_zprime_min, i_zprime_max = i_zprime_max, i_zprime_min

        res = [[], []]
        # Iterates along the Z' direction
        for iter_zprime in range(i_zprime_min, i_zprime_max + 1):
            # Sum of the volume values sampled from the X'Y' plane
            f_zprime = 0.

            # P_Z' = O' + (t * Z')
            p_zprime = start_p + z_prime * iter_zprime * delta_wrd
            # print(p_zprime)

            # We need to find a range along the Y' direction on which to perform
            # the integration. To do so, we computes the intersection between
            # the bounding box of the grid and the ray P_Y' = P_Z' + (t * Y').
            y_prime_aabb_inter, t_yprime_min, t_yprime_max = bbox.intersray(p_zprime, y_prime)

            # Express t_yprime_min and t_yprime_max as multiples of delta.
            # The result is rounded up for t_yprime_min and rounded down for
            # t_yprime_max, so to be sure to remain inside the grid.
            i_yprime_min, i_yprime_max = 0, 0
            if y_prime_aabb_inter:
                i_yprime_min = int(np.ceil(t_yprime_min / delta_wrd))
                i_yprime_max = int(np.floor(t_yprime_max / delta_wrd))
                if i_yprime_min > i_yprime_max:
                    i_yprime_max, i_yprime_min = i_yprime_min, i_yprime_max

            # Iterates along the Y' direction
            # Here caffeine parallelization!! TODO
            # pragma omp parallel for default(none), reduction(+:f_zprime),
            # schedule(dynamic), shared(i_yprime_min,i_yprime_max,P_zprime,
            # y_prime,deltaWorld,x_prime,worldToLocalBohr), firstprivate(bbox)
            for iter_yprime in range(i_yprime_min, i_yprime_max + 1):
                # P_Y' = P_Z' + (t * Y')
                p_yprime = p_zprime + y_prime * iter_yprime * delta_wrd
                # print(p_yprime)

                # We need to find a range along the X' direction on which to perform
                # the integration. To do so, we compute the intersection between
                # the bounding box of the grid and the ray.
                x_prime_aabb_inter, t_xprime_min, t_xprime_max =         bbox.intersray(p_yprime, x_prime)

                # Express t_xprime_min and t_xprime_max as multiples of delta.
                # The result is rounded up for t_xprime_min and rounded down for
                # t_xprime_max, so to be sure to remain inside the grid.
                i_xprime_min, i_xprime_max = 0, 0
                if x_prime_aabb_inter:
                    i_xprime_min = int(np.ceil(t_xprime_min / delta_wrd))
                    i_xprime_max = int(np.floor(t_xprime_max / delta_wrd))
                    if i_xprime_min > i_xprime_max:
                        i_xprime_min, i_xprime_max = i_xprime_max, i_xprime_min

                # Iterates along the X' direction
                for iter_xprime in range(i_xprime_min, i_xprime_max + 1):
                    # P' = P_X' = P_Y' + (t * X').
                    p_prime = p_yprime + x_prime * iter_xprime * delta_wrd
                    # print(p_prime)

                    # Computes P' in the local reference frame of the grid
                    p_prime_local = self._wrdtolocal(p_prime)

                    # Makes sure that P' lies within the cube
                    p_prime_local[0] = np.clip(p_prime_local[0], 0.,
                                               float(self.npts[0]-1))
                    p_prime_local[1] = np.clip(p_prime_local[1], 0.,
                                               float(self.npts[1]-1))
                    p_prime_local[2] = np.clip(p_prime_local[2], 0.,
                                               float(self.npts[2]-1))
                    # Sample the value of P_prime_local via linear interpolation
                    f_zprime += self._get_value(p_prime_local)

            t_zprime_world = delta_wrd * iter_zprime

            # Area of every sampling cell.
            cellarea = delta_wrd**2
            f_zprime *= cellarea

            # Store the current value of the function
            res[0].append(t_zprime_world)
            res[1].append(f_zprime)

        # If i_zprime_min and i_zprime_max have been swapped, the vector
        # of the results have to be reversed to be sorted.
        if (t_zprime_min > t_zprime_max):
            res = [list(reversed(res[0])),
                   list(reversed(res[1]))]
        return res




class VecCubeData(CubeData):
    """
    CubeData containing Vector fields (3d vec)
    """
    def __init__(self, cubedata: CubeData) -> None:
        try:
            if not cubedata.cube.shape[0] == 3:
                raise NoValidData('VecCubeData', 'Not vec field data set')
            CubeData.__init__(self, cubedata)
            self._getminmax()
        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

    def _getminmax(self) -> None:
        normvals = np.sqrt(np.einsum('ij,ij->j', self.cube, self.cube))
        self._minnorm = normvals.min()
        self._maxnorm = normvals.max()
        self._minind = normvals.argmin()
        self._maxind = normvals.argmax()
        self._minval = self.cube[:, self._minind]
        self._maxval = self.cube[:, self._maxind]

    def integrate(self, mask: np.ndarray | None = None) -> np.ndarray:
        """integrate vector field in all space
        param mask: boolen mask
        # TODO check the dimensions
        """
        if mask is not None:
            mask.any()
            vec_1 = self.cube[:, mask]
        else:
            vec_1 = self.cube
        poligon = self.get_voxvol()
        integrated = vec_1.sum(axis=1)*poligon
        return integrated

    def _calc_rot(self, mask: np.ndarray | None = None, origin: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the rotor of a a vector fields
        """
        if not self.box.size:
            self.make_box()
        if mask is not None:
            vec_1 = self.cube[:, mask]
            box_1 = self.box[:, mask]
        else:
            vec_1 = self.cube
            box_1 = self.box
        if origin is not None:
            box_1 -= origin[:, np.newaxis]
        if GPU:
            res = np.zeros_like(vec_1)
            for i_th in range(vec_1.shape[1]):
                cross3(box_1[:, i_th], vec_1[:, i_th], res[:, i_th])
        else:
            res = np.cross(box_1, vec_1, axisa=0, axisb=0).T

        return res

    def rotorintegrate(self, mask: np.ndarray | None = None, origin: np.ndarray | None = None) -> np.ndarray:
        rotfield = self._calc_rot(mask, origin=origin)
        poligon = self.get_voxvol()
        return rotfield.sum(axis=1)*poligon

    def proj_on_vec(self, vec: np.ndarray | None = None,
                   rot: bool = False, cube: bool = False) -> np.ndarray | CubeData:
        """
        Return a scalar cube with the projection
        of the ith elements
        contribution on a vector, if no vector is provided
        the integration of the grid is used
        return scalar field or a cube with the scalar fied if cube=True
        """
        if vec is None:
            vec = self.integrate()
        else:
            vec = np.array(vec)
            # BUG check if are floats!!
            if vec.shape != (3,):
                raise NoValidData(vec, 'vec must be a 3d vector')
        if rot:
            mat = self._calc_rot()
        else:
            mat = self.cube
        tmp = np.dot(vec, mat)
        if not cube:
            return tmp
        tmp2 = CubeData(self)
        tmp2.cube = tmp * 0.5
        tmp2.nval = 1
        return tmp2

    def get_norm(self, cube: bool = False) -> np.ndarray | CubeData:
        """
        Returns the norm of the vector fields in a CubeData if
        cube=True else retur the vector as np.array
        """
        res = np.sqrt(np.einsum('ij,ij->j', self.cube, self.cube))
        if not cube:
            return res
        tmp2 = CubeData(self)
        tmp2.cube = res
        tmp2.nval = 1
        return tmp2


class VtcdData(VecCubeData):
    """
    CubeData containing Vector fields from vtcd
    """
    def __init__(self, cubedata: CubeData, ithevec: np.ndarray, nu_freq: float) -> None:
        try:
            if not cubedata.cube.shape[0] == 3:
                raise NoValidData('VtcdData', 'Not vec field data set')
            VecCubeData.__init__(self, cubedata)
            self.evec = ithevec
            self.energy = nu_freq

        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))

    def mu_integrate(self, mask: np.ndarray | None = None) -> np.ndarray:
        """integrate vector field in all space"""
        integrated = self.integrate(mask=mask) * -1 * 2
        return integrated

    def mag_integrate(self, mask: np.ndarray | None = None, origin: np.ndarray | None = None) -> np.ndarray:
        """integrate vector field in all space"""
        poligon = self.get_voxvol()
        res = self._calc_rot(mask, origin=origin)
        # m = -1/2 * sum(cross(r,J)) in atomic units
        # integrated = res.sum(axis=1) * poligon / (2 * 137) * -1
        integrated = res.sum(axis=1) * poligon * -1/2
        return integrated

    def mag_integrate_test(self, mask: np.ndarray | None = None) -> np.ndarray:
        """multi origin?

        Args:
            mask (bool, optional): _description_. Defaults to False.
        """
        def cross(a, b):
            if GPU:
                res = np.zeros_like(b)
                if a.shape == b.shape:
                    for i_th in range(b.shape[1]):
                        cross3(a[:, i_th], b[:, i_th], res[:, i_th])
                elif a.shape[0] == 3:
                    for i_th in range(b.shape[1]):
                        cross3(a, b[:, i_th], res[:, i_th])
                else:
                    raise NotImplementedError
            else:
                res = np.cross(a, b, axisa=0, axisb=0).T
            return res
        if not self.box.size:
            self.make_box()
        if mask is not None:
            mask.any()
            vec_1 = self.cube[:, mask]
            box_1 = self.box[:, mask]
        else:
            vec_1 = self.cube
            box_1 = self.box
        poligon = self.get_voxvol()
        # m = -ie/2c sum_j(Rj+rj) cross J(r)
        res = np.zeros_like(vec_1, dtype=float)
        for i in range(self.natoms):
            res += cross(self.crd[i, :], vec_1)
            res += cross(box_1 - self.crd[i, :][:, np.newaxis], vec_1)
        integrated = res.sum(axis=1) * poligon * -1
        # integrated = res.sum(axis=1) * poligon / (2 * 137) * -1
        return integrated/self.natoms

    def mu_nuc(self) -> np.ndarray:
        """
        return the nuclear contribution to EDTM
        """
        pre_fc = 1
        return pre_fc * np.sum(self.evec.reshape(self.natoms, 3) *
                               np.expand_dims(self.ian, axis=1), axis=0)

    def mg_nuc(self) -> np.ndarray:
        """
        returns the nuclear contribution to MDTM
        """
        # pre_fc = 1/(2*137)
        pre_fc = 1
        tens = np.zeros((self.natoms*3, 3))
        for i in range(self.natoms):
            tens[i*3:i*3+3, :] = np.array([[0, -self.crd[i, 2], self.crd[i, 1]],
                                           [self.crd[i, 2], 0, -self.crd[i, 0]],
                                           [-self.crd[i, 1], self.crd[i, 0], 0]]).T*self.ian[i]/4
        res = np.dot(self.evec, tens)
        # print(tens)
        return pre_fc * res

    def proj_on_vec(self, typ: str = 'moe', nucl: bool = False, cube: bool = False) -> np.ndarray | CubeData:
        """
        compute the scalar field, projecting the vector field
        on the electric dipole transition moment
        params nucl: include the nuclear contribution
        """
        if typ not in ['moe', 'eom', 'eoe', 'mom']:
            raise NoValidData('VtcdData.proj_on_vec', 'not available')
        if typ == 'moe':
            vec = self.mu_integrate()
            if nucl:
                nuc = self.mu_nuc()
                vec += nuc
            rot = True
            pre_fc = 1
        elif typ == 'eom':
            vec = self.mag_integrate()
            if nucl:
                nuc = self.mg_nuc()
                vec += nuc
            rot = False
            pre_fc = 0.25
        elif typ == 'eoe':
            vec = self.mu_integrate()
            if nucl:
                nuc = self.mu_nuc()
                vec += nuc
            rot = False
            pre_fc = 0.25
        elif typ == 'mom':
            vec = self.mag_integrate()
            if nucl:
                nuc = self.mg_nuc()
                vec += nuc
            rot = True
            pre_fc = 1
        else:
            raise NotImplementedError
        # *-1 is to match the direction of the vectors in the cube
        res = super(VtcdData, self).proj_on_vec(vec=vec*-1, rot=rot, cube=cube)
        if cube:
            res.cube /= pre_fc
        else:
            res /= pre_fc
        return res


class AimCubeData(VecCubeData):
    """
    cube data masked with AIM partition
    of the space
    """
    def __init__(self, cubedata: VecCubeData | VtcdData,
                 mask: CubeData) -> None:
        if not isinstance(mask, CubeData):
            raise NoValidData('AimCubeData', 'mask: CubeData object required')
        if not isinstance(cubedata, VecCubeData) and not isinstance(cubedata, VtcdData):
            raise NoValidData('AimCubeData', 'VecCubeData or VtcdData object required')
        if not cubedata.same_system(mask):
            raise NoValidData("AimCubeData", """Data set and mask did not
        share the same volumetric space""")
        if isinstance(cubedata, VtcdData):
            VtcdData.__init__(self, cubedata, cubedata.evec, cubedata.energy)
        elif isinstance(cubedata, VecCubeData):
            VecCubeData.__init__(self, cubedata)
        else:
            raise NoValidData("AimCubeData", "Not valid data type")
        self.mask = mask.cube
        self.basin = self.__get_atom_voxel()
        self._frag = None
        if not self.box.size:
            self.make_box()

    def __get_atom_voxel(self) -> list[int]:
        """
        define a dictionary entry beetween the actractora and the atoms
        """
        atom_multimap = []
        total_voxels = self.npts[0] * self.npts[1] * self.npts[2]
        
        for i_th in range(self.natoms):
            local_atom = [int(x) for x in np.dot(np.linalg.inv(self.loc2wrd),
                                                 np.append(self.crd[i_th], 1))[:3]]
            index = local_atom[0]*self.npts[1]*self.npts[2] +\
                    local_atom[1]*self.npts[2] + local_atom[2]
                    
            # Add bounds checking to prevent IndexError
            if index < 0 or index >= total_voxels:
                print(f"Warning: Atom {i_th} index {index} is out of bounds [0, {total_voxels-1}]")
                print(f"  local_atom coords: {local_atom}")
                print(f"  world coords: {self.crd[i_th]}")
                print(f"  npts: {self.npts}")
                # Use a default value or skip this atom
                atom_multimap.append(0)  # Default to basin 0
                continue
                
            # Additional check for mask array dimensions
            if hasattr(self.mask, 'shape'):
                if self.mask.ndim == 1 and index >= len(self.mask):
                    print(f"Warning: Index {index} exceeds mask length {len(self.mask)}")
                    atom_multimap.append(0)
                    continue
                elif self.mask.ndim == 2 and index >= self.mask.shape[1]:
                    print(f"Warning: Index {index} exceeds mask second dimension {self.mask.shape[1]}")
                    atom_multimap.append(0)
                    continue
                    
            atom_multimap.append(self.mask[index])
        return atom_multimap

    def get_atom_mask(self, atom_id: int) -> np.ndarray:
        """
        return a bool 1d array, true for the vouxel
        assigned to the atom_id
        """
        return self.mask == self.basin[atom_id]

    def set_fragments(self, frags: list[list[int]]) -> None:
        """
        add the fragments
        """
        # shift the indeces
        # sfrags = [[x-1 for x in item] for item in frags]
        # no shift, handled outside
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
                self._frag = None
                print(unic_atms)
                print(sfrags)
                print(frags)
                raise ValueError
            else:
                excluded = list(set(range(self.natoms)) - set(tmp_full))
                if excluded:
                    sfrags.append(excluded)
                self._frag = [tuple(x) for x in sfrags]
        except Exception as err:
            print(err)

    def get_atom_contribution(self, atom_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        return two vetor with the atomic contribution to
        EDTM and MDTM
        """
        mask = self.get_atom_mask(atom_id)
        return self.integrate(mask), self.rotorintegrate(mask)

    def get_domag(self) -> np.ndarray:
        def cross(a, b):
            if GPU:
                res = np.zeros_like(b)
                if a.shape == b.shape:
                    for i_th in range(b.shape[1]):
                        cross3(a[:, i_th], b[:, i_th], res[:, i_th])
                elif a.shape[0] == 3:
                    for i_th in range(b.shape[1]):
                        cross3(a, b[:, i_th], res[:, i_th])
                else:
                    raise NotImplementedError
            else:
                res = np.cross(a, b, axisa=0, axisb=0).T
            return res
        
        if not self.box.size:
            self.make_box()

        poligon = self.get_voxvol()
        # m = -ie/2c sum_j(Rj+rj) cross J(r)
        res = np.zeros(3)
        for i in range(self.natoms):
            mask = self.get_atom_mask(i)
            vec_1 = self.cube[:, mask]
            box_1 = self.box[:, mask]
            tmp = cross(self.crd[i, :], vec_1)
            tmp += cross(box_1 - self.crd[i, :][:, np.newaxis], vec_1)
            res += tmp.sum(axis=1) * poligon * -1
        return res

    def mu_integrate(self, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Integration for electric dipole moment
        Delegates to appropriate method based on the underlying data type
        """
        if hasattr(self, 'evec') and hasattr(self, 'energy'):
            # This is VTCD data, use specialized integration
            integrated = self.integrate(mask=mask) * -1 
            return integrated
        else:
            # Regular vector field integration
            return self.integrate(mask=mask)

    def mag_integrate(self, mask: np.ndarray | None = None, origin: np.ndarray | None = None) -> np.ndarray:
        """
        Integration for magnetic dipole moment
        Delegates to appropriate method based on the underlying data type
        """
        if hasattr(self, 'evec') and hasattr(self, 'energy'):
            # This is VTCD data, use specialized integration
            poligon = self.get_voxvol()
            res = self._calc_rot(mask, origin=origin)
            integrated = res.sum(axis=1) * poligon * -1/2
            return integrated
        else:
            # Regular rotational integration
            return self.rotorintegrate(mask=mask, origin=origin)

    def get_frags_contribution(self) -> dict[str, list[np.ndarray]]:
        if self._frag is None:
            raise NoValidData('get_frags_contribution', 'Fragments not set')
        
        print("\n=== DEBUG: Fragment Contribution Analysis ===")
        
        res = {'int': [],
               'rot': []}
        
        # Debug: Compute total integration without fragments for comparison
        total_int = self.mu_integrate()
        total_rot = self.mag_integrate()
        print("Total integration (no fragments):")
        print(f"  Electric: {total_int}")
        print(f"  Magnetic: {total_rot}")
        print()
        
        # Process each fragment
        for i, frg in enumerate(self._frag):
            mask_tmp = np.full(self.mask.shape, False)
            if isinstance(frg, tuple):
                atoms_in_frag = list(frg)
                for atom in frg:
                    mask_tmp += self.get_atom_mask(atom)
            else:
                atoms_in_frag = [frg]
                mask_tmp = self.get_atom_mask(frg)
            
            frag_int = self.mu_integrate(mask_tmp)
            frag_rot = self.mag_integrate(mask_tmp)
            
            res['int'].append(frag_int)
            res['rot'].append(frag_rot)
            
            # Debug: Print each fragment's contribution
            print(f"Fragment {i+1} (atoms {[a+1 for a in atoms_in_frag]}):")
            print(f"  Electric: {frag_int}")
            print(f"  Magnetic: {frag_rot}")
        
        # Debug: Compute and display sum of fragments
        sum_int = np.sum(res['int'], axis=0)
        sum_rot = np.sum(res['rot'], axis=0)
        print()
        print("Sum of all fragments:")
        print(f"  Electric: {sum_int}")
        print(f"  Magnetic: {sum_rot}")
        
        # Debug: Show difference between total and sum of fragments
        diff_int = total_int - sum_int
        diff_rot = total_rot - sum_rot
        print()
        print("Difference (Total - Sum of fragments):")
        print(f"  Electric: {diff_int}")
        print(f"  Magnetic: {diff_rot}")
        print("============================================\n")
        
        return res

    def get_frag_isosurf(self) -> CubeData:
        if self._frag is None:
            raise NoValidData('get_frag_isosurf', 'Fragments not set')
        tmpcube = CubeData(self)
        tmpcube.nval = len(self._frag)
        relpos = np.array([[-1, 0, 0],
                           [ 1, 0, 0],
                           [ 0,-1, 0],
                           [ 0, 1, 0],
                           [ 0, 0,-1],
                           [ 0, 0, 1]])
        mask_brd = np.full((len(self._frag), self.mask.shape[0]), 0) 
        for i, frg in enumerate(self._frag):
            mask_tmp = np.full(self.mask.shape, False)
            if isinstance(frg, tuple):
                for atom in frg:
                    mask_tmp += self.get_atom_mask(atom)
            else:
                mask_tmp = self.get_atom_mask(frg)
            mask_indices = np.where(mask_tmp)[0]
            for indx in mask_indices:
                xyz = self._lineartocube(indx)
                if (xyz[0] == 0 or xyz[1] == 0 or xyz[2] == 0 or 
                   xyz[0] == self.npts[0] or xyz[1] == self.npts[1] or
                   xyz[2] == self.npts[2]):
                    mask_brd[i, indx] = 1
                else:
                    xyz_rel = relpos + np.array(xyz)
                    # Check bounds for all relative positions before calculating indices
                    valid_positions = []
                    rel_indx = []
                    for pos in xyz_rel:
                        if (pos[0] >= 0 and pos[0] < self.npts[0] and
                            pos[1] >= 0 and pos[1] < self.npts[1] and
                            pos[2] >= 0 and pos[2] < self.npts[2]):
                            rel_indx.append(self._cubetolinear(pos))
                            valid_positions.append(True)
                        else:
                            valid_positions.append(False)
                    
                    # Only check mask if we have valid indices and all positions are valid
                    if len(rel_indx) == len(xyz_rel) and all(valid_positions):
                        if mask_tmp[rel_indx].all():
                            mask_brd[i, indx] = 1
        tmpcube.cube = mask_brd
        return tmpcube
             
    # def get_groups_contribution(self, groups):
    #     """
    #     Return the contribution of the selected groups
    #     passed as string, the string is parsed as with
    #     range_parse. All the elements should are then scaled by one to
    #     be transformed to indexes
    #     """
    #     grp_rd = range_parse(groups, dimension=self.natoms)
    #     data_tmp = []
    #     mu_tot = self.mu_integrate()
    #     for grp in grp_rd:
    #         mask_tmp = np.full(self.mask.shape, False)
    #         if isinstance(grp, list):
    #             for sub_grp in grp:
    #                 mask_tmp += self.get_atom_mask(sub_grp-1)
    #         else:
    #             mask_tmp = self.get_atom_mask(grp-1)
    #         data_tmp.append((np.dot(mu_tot, self.mu_integrate(mask_tmp)),
    #                          np.dot(mu_tot, self.mag_integrate(mask_tmp))))
    #     return data_tmp


def print_cube(cubdata: 'CubeData', fname: str = 'cubefile.cube',
               comment: str = 'Commentline',
               vec_pr: np.ndarray | None = None, gau_style: bool = True) -> None:
    """
    Print a volumetric dataset contained in CubeData
    as Gaussian Cube File
    param cubedata: Cubedata object containing the system information
    param fname: name of the output file
    param comment: string to be include as comment in the file
    param vec_pr: a ndarray to be write instead of cubedata.cube
    param gau_style: add a \\n after each z index end
    """
    outfile = '{}'.format(fname)
    fout = open(outfile, 'w', encoding='utf-8')
    fout.write(' Cube generated with tcdlib\n')
    fout.write(' {}\n'.format(comment))
    fmt_2 = '{:5d}{b[0]:12.6f}{b[1]:12.6f}{b[2]:12.6f}'
    fmt_3 = '{a:5d}{a:12.6f}{b[0]:12.6f}{b[1]:12.6f}{b[2]:12.6f}\n'
    fout.write(fmt_2.format(cubdata.natoms,
                            b=cubdata.loc2wrd[:3, 3])+'{:5d}\n'.format(cubdata.nval))
    for i in range(3):
        fout.write(fmt_2.format(cubdata.npts[i],
                                b=cubdata.loc2wrd[:3, i])+'\n')
    npts = cubdata.npts[0] * cubdata.npts[1] * cubdata.npts[2]
    for j in range(cubdata.natoms):
        fout.write(fmt_3.format(a=int(cubdata.ian[j]),
                                b=cubdata.crd[j, :]))
    if vec_pr is None:
        vec_pr = cubdata.cube
    elif isinstance(vec_pr, np.ndarray):
        if vec_pr.shape[0] != npts:
            raise NoValidData('print_cube',
                              '{} points expected, {} present in the array'.format(npts, vec_pr.shape[0]))
        elif vec_pr.shape != cubdata.cube.shape:
            raise NoValidData('print_cube',
                              'vec_pr and cubdata.cube have different number of Nval. {} expected.'.format(cubdata.nval))
    else:
        raise NoValidData('print_cube', 'ndarray expected')
    # Check if it is a one dimension array
    try:
        vec_pr.shape[1]
    except IndexError:
        vec_pr = np.expand_dims(vec_pr, axis=1)
    count = 0

    for i in range(vec_pr.shape[1]):
        for j in range(vec_pr.shape[0]):
            count += 1
            fout.write(' {:12.5E}'.format(vec_pr[j, i]))
            if count % 6 == 0:
                fout.write("\n")
            elif gau_style and count % cubdata.npts[2] == 0:
                fout.write("\n")
                count = 0

    fout.close()


# def print_vec_incube(fname, vect1, vect2):
#     """
#     Hack to print EDTM and MDTM into a cube file
#     """
#     dcube = CubeData()
#     dcube.natoms = 0
#     dcube.loc2wrd = np.identity(4)
#     dcube.npts = [2, 2, 2]
#     dcube.nval = 3
#     dcube.cube = np.zeros((3, 8))
#     dcube.cube[:, 0] = vect1
#     dcube.cube[:, 1] = vect2
#     print_cube(dcube, 'ELC+MAG', '{}'.format(fname))


def cube_parser(cubfile: str,
                elemenents: bool = True) -> CubeData:
    """
    Read and extract data from a cube
    """
    if not os.path.exists(cubfile):
        print('ERROR: Cube file "{}" not found'.format(cubfile))
        raise OSError

    data_tmp = CubeData()
    with open(cubfile, 'r', encoding='utf-8') as fi_le:
        # 1st 2 lines are titles/comments. Ignored for now
        line = fi_le.readline()  # Title
        line = fi_le.readline()  # Comment
        line = fi_le.readline()  # NAtoms, X0, Y0, Z0, NVal
        token = line.split()
        mo_cube = False
        if int(token[0]) < 0:
            mo_cube = True
        data_tmp.set_natoms(np.abs(int(token[0])))
        data_tmp.loc2wrd[:3, 3] = [float(e) for e in token[1:4]]
        if len(token) > 4:
            nval = int(token[4])
        else:
            nval = 1
        data_tmp.nval = nval
        # REMINDER: Gaussian does not require the grid to be "rectangular"
        # N1, X1, Y1, Z1 (displacement along 1st coord)
        for i_th in range(3):
            line = fi_le.readline()
            token = line.split()
            data_tmp.npts[i_th] = int(token[0])
            data_tmp.loc2wrd[:3, i_th] = [float(e) for e in token[1:4]]
            data_tmp.wrd2loc = np.linalg.inv(data_tmp.loc2wrd)
        for i_th in range(data_tmp.natoms):
            line = fi_le.readline()
            # AtNum(i)x2, X(i), Y(i), Z(i)
            token = line.split()
            data_tmp.add_atom(int(token[0]), [float(x) for x in token[2:]], i_th)
        if mo_cube:
            nlines = ceil((data_tmp.nval+1)/10)
            data_tmp = ScalarsCube(data_tmp)
            labels = []
            for i_th in range(nlines):
                line = fi_le.readline()
                labels.extend([str(x) for x in line.split()])
            data_tmp.set_labels(labels[1:])

        content = fi_le.readlines()
    vec_num = []
    for line in content:
        vec_num.extend([float(x) for x in line.split()])
    vec_num = np.array(vec_num)
    if nval > 1:
        vec_num.resize(int(vec_num.shape[0] / nval), nval)
        vec_num = vec_num.transpose()
    if nval == 3 and not elemenents:
        # Correction between old behavior and elements
        vec_num *= np.sqrt(2)
    data_tmp.cube = vec_num

    return data_tmp

