#!/bin/python3
"""
Genenarl function to operate with vectors and matrices
"""
# from math import pi, sqrt
import os
import typing as tp
import numpy as np
# from .types import Array3F
from .custom_except import NoValidData, NoValidArg
from .mol_data import AT_RVDW

def fuzzy_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """
    Check if two floats are equal within a tolerance
    """
    return abs(a - b) < tol

def check_nx3array(vec, mod=True):
    """
    check if the provided listo or numpy array is
    a Nx3 matrix. if mod=True return the vec as
    numpy array in Nx3 format if a compatible is
    provided:
        list/tuple Nx3 or 3xN
        nparray Nx3 or 3xN
    """

    if isinstance(vec, (list, tuple, np.ndarray)):
        vec = np.array(vec)
    elif vec is None:
        raise AttributeError
    else:
        raise NoValidData("check_nx3array, vec:",
                          "Not supported format")
    if len(vec.shape) != 2:
        raise NoValidData("check_nx3array",
                          "vec should be a 2d array")
    if vec.shape[0] == 3 and vec.shape[1] != 3:
        if mod:
            print("Probably provided a 3xN array; trasposed")
            vec = vec.T
        else:
            raise NoValidData("check_nx3array",
                              "vec should be a Nx3 matrix")
    elif vec.shape[1] != 3:
        raise NoValidData("check_nx3array",
                          "vec should be a Nx3 matrix")

    return vec


def get_ivib(nquant, lmods, dmods, nvb):
    """
    return the variational state index required:
    nquant: number of quanta
    lmods: ordered list of modes
    dmods: dictionary with [mode]:number of quanta
    nvb: number of vibrational state
    """
    idvstat = 0
    if nquant == 1:
        idvstat = lmods[0]
    elif nquant == 2:
        idvstat = nvb
        i = lmods[0]
        if len(lmods) == 1:
            idvstat += i*(i+1)/2
        else:
            idvstat += i*(i-1)/2 + lmods[1]
    elif nquant == 3:
        idvstat = nvb + nvb*(nvb+1)/2
        i = lmods[0]
        if len(lmods) == 1:
            idvstat += i*(i+1)*(i+2)/6
        else:
            j = lmods[1]
            if len(lmods) == 2:
                idvstat += i*(i-1)*(i-2)/6 + j
                if dmods[i] == 2:
                    idvstat += i*(i-1)/2
                else:
                    idvstat += j*(j-1)/2
            else:
                idvstat += i*(i-1)*(i-2)/6 + j*(j-1)/2 + lmods[2]
    else:
        raise NoValidArg(nquant,
                         'ERROR: Unsupported number of quanta.')
    return idvstat


def get_vibstate(instr, nvib):
    """
    get a string of type 'n+m' or '2n' and returns the vibration
    index of the state and smodes
    """
    lvibosc = instr.split('+')
    dmodes = {}
    nquanta = 0
    for vibosc in lvibosc:
        if '*' in vibosc:
            data = vibosc.split('*')
            numq = int(data[0])
            mode = int(data[1])
        else:
            numq = 1
            mode = int(vibosc)
        nquanta += numq
        if mode > nvib:
            raise NoValidArg(mode,
                             'ERROR: Absolute variational state\
                             not yet available')
        if mode in dmodes:
            dmodes[mode] += numq
        else:
            dmodes[mode] = numq
    smodes = []
    lmodes = sorted(dmodes.keys(), reverse=True)
    for mode in lmodes:
        smodes.append('{}({})'.format(mode, dmodes[mode]))
    idvstate = get_ivib(nquanta, lmodes, dmodes, nvib)

    return idvstate, smodes


def get_connect(ian, coord):
    """
    return a list with the connectivity
    """
    coord = np.array(coord)
    atnum = coord.shape[0]
    # dist = np.zeros((atnum, atnum))
    connect = []
    for i in range(atnum):
        for j in range(atnum):
            if j >= i:
                break
            dist = np.sqrt((coord[i, 0] - coord[j, 0])**2 +
                           (coord[i, 1] - coord[j, 1])**2 +
                           (coord[i, 2] - coord[j, 2])**2)
            vdw_dist = AT_RVDW[int(ian[i])] + AT_RVDW[int(ian[j])]
            if dist < vdw_dist:
                connect.append((i+1, j+1))
    return connect

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(vc1, vc2):
    """ Returns the angle in radians between vectors 'vc1' and 'vc2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(vc1)
    v2_u = unit_vector(vc2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# https://stackoverflow.com/a/6802723
def rotation_matrix(vec0, vec1):
    """
    Rodrigues' rotation formula
    vec0 will match vec1
    """
    vec_0n = unit_vector(vec0)
    vec_1n = unit_vector(vec1)
    comp = abs(np.linalg.norm(vec_0n - vec_1n))
    ident = np.identity(3)
    if comp < 1e-6:
        r_mat = ident
    elif abs(2 - comp) < 1e-6:
        r_mat = ident * -1
    else:
        axis = unit_vector(np.cross(vec_0n, vec_1n))
        angle = np.arccos(np.dot(vec_0n, vec_1n))
        a_mat = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        r_mat = ident + np.sin(angle) * a_mat + (1 - np.cos(angle)) * np.dot(a_mat,
                                                                             a_mat)

    return r_mat


def range_parse(string, dimension=None, flatten=False):
    """
    Parse a string defining range.
    return a list of list
    Rules:
        , define groups
        + add number to the group
        - define a range (eg. -3 = [1,2,3])
        ! before a range state that each element is a groups
    Example:
        number of elements = 13
        "-2,3,4,5,7+9,8,!10-13"
        return: [[1,2],3,4,5,[7,9],8,10,11,12]
    """
    grp_elem = []
    for grp in string.split(','):
        if '-' not in grp and '+' not in grp:
            grp_elem.append(int(grp))
        elif '+' in grp:
            tmp = [int(x) for x in grp.split('+')]
            grp_elem.append(tmp)
        else:
            flag = False
            if grp.startswith('!'):
                grp = grp[1:]
                flag = True
            tmp = grp.split('-')
            try:
                strt = int(tmp[0])
            except ValueError:
                strt = 1
            try:
                end = int(tmp[1]) + 1
            except ValueError:
                if dimension is None:
                    raise NoValidData("range_parse", "end of the range not defined")
                else:
                    end = dimension
            toadd = [int(x) for x in range(strt, end)]
            if flag:
                grp_elem.extend(toadd)
            else:
                grp_elem.append(toadd)
    if flatten:
        flattened = []
        for item in grp_elem:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        grp_elem = flattened
    return grp_elem


class Streamlines(object):
    """
    Copyright (c) 2011 Raymond Speth.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    See: http://web.mit.edu/speth/Public/streamlines.py
    """

    def __init__(self, X, Y, U, V, res=0.125,
                 spacing=2, maxLen=2500, detectLoops=False):
        """
        Compute a set of streamlines covering the given velocity field.

        X and Y - 1D or 2D (e.g. generated by np.meshgrid) arrays of the
                  grid points. The mesh spacing is assumed to be uniform
                  in each dimension.
        U and V - 2D arrays of the velocity field.
        res - Sets the distance between successive points in each
              streamline (same units as X and Y)
        spacing - Sets the minimum density of streamlines, in grid points.
        maxLen - The maximum length of an individual streamline segment.
        detectLoops - Determines whether an attempt is made to stop extending
                      a given streamline before reaching maxLen points if
                      it forms a closed loop or reaches a velocity node.

        Plots are generated with the 'plot' or 'plotArrows' methods.
        """

        self.spacing = spacing
        self.detectLoops = detectLoops
        self.maxLen = maxLen
        self.res = res

        xa = np.asanyarray(X)
        ya = np.asanyarray(Y)
        self.x = xa if xa.ndim == 1 else xa[0]
        self.y = ya if ya.ndim == 1 else ya[:,0]
        self.u = U
        self.v = V
        self.dx = (self.x[-1]-self.x[0])/(self.x.size-1) # assume a regular grid
        self.dy = (self.y[-1]-self.y[0])/(self.y.size-1) # assume a regular grid
        self.dr = self.res * np.sqrt(self.dx * self.dy)

        # marker for which regions have contours
        self.used = np.zeros(self.u.shape, dtype=bool)
        self.used[0] = True
        self.used[-1] = True
        self.used[:,0] = True
        self.used[:,-1] = True

        # Don't try to compute streamlines in regions where there is no velocity data
        for i in range(self.x.size):
            for j in range(self.y.size):
                if self.u[j,i] == 0.0 and self.v[j,i] == 0.0:
                    self.used[j,i] = True

        # Make the streamlines
        self.streamlines = []
        while not self.used.all():
            nz = np.transpose(np.logical_not(self.used).nonzero())
            # Make a streamline starting at the first unrepresented grid point
            self.streamlines.append(self._makeStreamline(self.x[nz[0][1]],
                                                         self.y[nz[0][0]]))


    def _interp(self, x, y):
        """ Compute the velocity at point (x,y) """
        i = (x-self.x[0])/self.dx
        ai = i % 1

        j = (y-self.y[0])/self.dy
        aj = j % 1

        i, j = int(i), int(j)

        # Bilinear interpolation
        u = (self.u[j,i]*(1-ai)*(1-aj) +
             self.u[j,i+1]*ai*(1-aj) +
             self.u[j+1,i]*(1-ai)*aj +
             self.u[j+1,i+1]*ai*aj)

        v = (self.v[j,i]*(1-ai)*(1-aj) +
             self.v[j,i+1]*ai*(1-aj) +
             self.v[j+1,i]*(1-ai)*aj +
             self.v[j+1,i+1]*ai*aj)

        self.used[j:j+self.spacing,i:i+self.spacing] = True

        return u,v

    def _makeStreamline(self, x0, y0):
        """
        Compute a streamline extending in both directions from the given point.
        """

        sx, sy = self._makeHalfStreamline(x0, y0, 1) # forwards
        rx, ry = self._makeHalfStreamline(x0, y0, -1) # backwards

        rx.reverse()
        ry.reverse()

        return rx+[x0]+sx, ry+[y0]+sy

    def _makeHalfStreamline(self, x0, y0, sign):
        """
        Compute a streamline extending in one direction from the given point.
        """

        xmin = self.x[0]
        xmax = self.x[-1]
        ymin = self.y[0]
        ymax = self.y[-1]

        sx = []
        sy = []

        x = x0
        y = y0
        i = 0
        while xmin < x < xmax and ymin < y < ymax:
            u, v = self._interp(x, y)
            theta = np.arctan2(v,u)

            x += sign * self.dr * np.cos(theta)
            y += sign * self.dr * np.sin(theta)
            sx.append(x)
            sy.append(y)

            i += 1

            if self.detectLoops and i % 10 == 0 and self._detectLoop(sx, sy):
                break

            if i > self.maxLen / 2:
                break

        return sx, sy

    def _detectLoop(self, xVals, yVals):
        """ Detect closed loops and nodes in a streamline. """
        x = xVals[-1]
        y = yVals[-1]
        D = np.array([np.hypot(x-xj, y-yj)
                      for xj,yj in zip(xVals[:-1],yVals[:-1])])
        return (D < 0.9 * self.dr).any()


def print_lines(lines_to_prt, state, vib, fold):
    """
    Stupid function to print a formatted line
    """
    ofile = open(os.path.join(fold, 'dipolePrint_VibState{}_{}.dat'.format(vib, state)), 'w', encoding='utf-8')
    ofile.write('# {}    {}    {}\n'.format('N', 'EDTM', 'MDTM'))
    for line in lines_to_prt:
        ofile.write(line)
    ofile.close()

def rayplaneintersection(rayori, raydir, planepoint, planenormal):
    npl = planenormal / (np.sqrt(np.dot(planenormal, planenormal)))
    ndotd = np.dot(npl, raydir)
    # If the ray is (almost) parallel to the plane, there is no intersection
    eps = 1e-5
    if np.abs(ndotd) < eps:
        return (False, np.nan)
    out_t = np.dot(planepoint - rayori, npl) / ndotd
    return (True, out_t)


class Slab():
    """
    TODO
    """

    def __init__(self, 
                 pointplane: np.ndarray = np.array([0, 0, -1]),
                 normalplane: np.ndarray = np.array([0, 0, -1]),
                 distplane: float = 2.0):
        self._set(pointplane, normalplane, distplane)

    def _set(self, pointplane,
              normalplane, distplane):
        eps = 1e-5
        if not non_nul_val(normalplane, eps):
            raise NoValidData('Slab.__set',
                              'invalid normal vector!')
        if distplane < eps:
            raise NoValidData('Slab.__set',
                              'invalid distance!')
        self.__refpoint = pointplane
        self.__normal = normalplane / (np.sqrt(np.dot(normalplane, normalplane)))
        self.__plndis = distplane

    def _getplane1(self):
        return (self.__refpoint, self.__normal)

    def _getplane2(self):
        tmp = (-self.__plndis * self.__normal) + self.__refpoint
        return (tmp, -1 * self.__normal)

    def isinside(self, point: np.ndarray,
                 epsilon: float = 1e-5) -> bool:
        tmp = np.dot(-1 * self.__normal, point - self.__refpoint)
        return (tmp - self.__plndis) <= epsilon <= tmp

    def interray(self,
                 rayori: np.ndarray,
                 raydir: np.ndarray):
        res, out_t1 = rayplaneintersection(rayori, raydir, *self._getplane1())
        if not res:
            return (False, out_t1, np.nan)
        res, out_t2 = rayplaneintersection(rayori, raydir, *self._getplane2())
        return (res, out_t1, out_t2)


def non_nul_val(vec: np.ndarray, tresh: float) -> bool:
    def lensq(a):
        return np.dot(a, a)
    if lensq(vec) < tresh**2:
        return False
    return True

Restype = tp.Tuple[bool, float, float]

class Poliedron():
    """
    Class that contains helper functions
    """

    def __init__(self,
                 origin: np.ndarray = np.array([0, 0, 0]),
                 x_extent: np.ndarray = np.array([1, 0, 0]),
                 y_extent: np.ndarray = np.array([0, 1, 0]),
                 z_extent: np.ndarray = np.array([0, 0, 1])) -> None:
        self.__origin = origin
        self._setextent(x_extent, y_extent, z_extent)

    def _setorigin(self, origin: np.ndarray) -> None:
        self.__origin = origin
        self._update_slab()

    def _setextent(self,
                   x_extent: np.ndarray,
                   y_extent: np.ndarray,
                   z_extent: np.ndarray) -> None:
        eps = 1e-5
        if (not non_nul_val(x_extent, eps) or
            not non_nul_val(y_extent, eps) or
            not non_nul_val(z_extent, eps)):
            raise NoValidData('Poligon._setextent',
                              'Parallelepiped: invalid extent vector!')
        self.__extent = [x_extent, y_extent, z_extent]
        self._update_slab()

    def _update_slab(self) -> None:
        """
        Set the extension of the parallelepiped by means of three vectors
        (representing the parallelepiped local axes).

        throw NoValidData if one of more vectors passed as parameters have length almost equals to 0.
        """
        def lun(a):
            return np.sqrt(np.dot(a, a))
        self.__slab = []
        for i in range(3):
            self.__slab.append(Slab(self.__origin, -1*self.__extent[i],
                               lun(self.__extent[i])))
    
    def getorigin(self) -> np.ndarray:
        return self.__origin

    def getxvec(self) -> np.ndarray:
        return self.__extent[0]

    def getyvec(self) -> np.ndarray:
        return self.__extent[1]

    def getzvec(self) -> np.ndarray:
        return self.__extent[2]

    def isinside(self, point: np.ndarray) -> bool:
        return (self.__slab[0].isinside(point) and
                self.__slab[1].isinside(point) and
                self.__slab[2].isinside(point))

    def intersray(self, rayori: np.ndarray,
                  raydir: np.ndarray) -> Restype:
        outtmin = -np.inf
        outtmax = np.inf
        tmp_tmin = 0
        tmp_tmax = 0
        for i in range(3):
            res, tmp_tmin, tmp_tmax = self.__slab[i].interray(rayori, raydir)
            if not res:
                if not self.__slab[i].isinside(rayori):
                    return (False, tmp_tmin, tmp_tmax)
            else:
                if tmp_tmin > tmp_tmax:
                    tmp_tmin, tmp_tmax = tmp_tmax, tmp_tmin
                outtmin = np.maximum(tmp_tmin, outtmin)
                outtmax = np.minimum(tmp_tmax, outtmax)
                if outtmin > outtmax:
                    return (False, np.nan, np.nan)
        return (res, outtmin, outtmax)


def trilinerarinterpolation(cube_values,
                            norm_coords):
    """
    @brief  Given 8 values associated to the vertices of a cube and
                     the normalized coordinates of a point within the cube,
                     this function computes an approximated value for the
                     specified point by interpolating linearly the values of
                     the vertices of the cube.

    @param       cube_values      Array containing the values associated to the 8
                                                 vertices of the cube. The values must be provided
                                                 in the following order:
                                                 {V000, V001, V010, V011, V100, V101, V110, V111}
    @param       norm_coords      Normalized coordinates of a point within the cube.
                                                 Must be in the range [0,1]. If not, they will be
                                                 claped to that range.
    @note        The type parameter T must implement the binary sum operator
                     (T+T) and the multiplication with a scalar value (double*T).
"""
    # Clamps the provided coordinates in the range [0,1]
    norm_coordsx = np.clip(norm_coords[0], 0., 1.)
    norm_coordsy = np.clip(norm_coords[1], 0., 1.)
    norm_coordsz = np.clip(norm_coords[2], 0., 1.)

    # Little optimization: precomputes 1-norm_coordsX/Y/Z
    norm_coords_negx = 1.0-norm_coordsx
    norm_coords_negy = 1.0-norm_coordsy
    norm_coords_negz = 1.0-norm_coordsz
    #
    # Computes and returns the tri-linear interpolation of the cube's values
    #
    # Vxyz =       V000 (1 - x) (1 - y) (1 - z) +
    #              V001 (1 - x) (1 - y)    z    +
    #              V010 (1 - x)    y    (1 - z) +
    #              V011 (1 - x)    y       z    +
    #              V100    x    (1 - y) (1 - z) +
    #              V101    x    (1 - y)    z    +
    #              V110    x       y    (1 - z) +
    #              V111    x       y       z
    #
    return (((norm_coords_negx * norm_coords_negy * norm_coords_negz) * cube_values[0]) +
           ((norm_coords_negx * norm_coords_negy * norm_coordsz) * cube_values[1]) +
           ((norm_coords_negx * norm_coordsy * norm_coords_negz) * cube_values[2]) +
           ((norm_coords_negx * norm_coordsy * norm_coordsz) * cube_values[3]) +
           ((norm_coordsx * norm_coords_negy * norm_coords_negz) * cube_values[4]) +
           ((norm_coordsx * norm_coords_negy * norm_coordsz) * cube_values[5] ) +
           ((norm_coordsx * norm_coordsy * norm_coords_negz) * cube_values[6] ) +
           ((norm_coordsx * norm_coordsy * norm_coordsz) * cube_values[7]
           ))

def sample(smatrix, z_hat, m_FA, Gamma_Threshold=1.0) -> np.ndarray:
    """Sample points from a multivariate normal distribution 
    (the ellipsoid enclosing the molecular system)
    Taken from:
    https://math.stackexchange.com/questions/2174751/generate-random-points-within-n-dimensional-ellipsoid
    https://www.onera.fr/sites/default/files/297/C013_-_Dezert_-_YBSTributeMonterey2001.pdf

    Args:
        smatrix (_type_): covariance matrix
        z_hat (_type_): origin of the distribution (ellipse center)
        m_FA (_type_): number of points to sample
        Gamma_Threshold (float, optional): _description_Gating threshold (>0). Defaults to 1.0.

    Returns:
        np.ndarray: (3xm_FA) array with the sampled points (3D coordinates
    """

    nz = smatrix.shape[0]
    z_hat = z_hat.reshape(nz, 1)

    X_Cnz = np.random.normal(size=(nz, m_FA))

    rss_array = np.sqrt(np.sum(np.square(X_Cnz),axis=0))
    kron_prod = np.kron( np.ones((nz,1)), rss_array)

    X_Cnz = X_Cnz / kron_prod       # Points uniformly distributed on hypersphere surface

    R = np.ones((nz,1))*( np.power( np.random.rand(1,m_FA), (1./nz)))

    unif_sph=R*X_Cnz               # m_FA points within the hypersphere
    T = np.asmatrix(np.linalg.cholesky(smatrix))    # Cholesky factorization of S => S=T’T


    unif_ell = T.H*unif_sph  # Hypersphere to hyperellipsoid mapping

    # Translation and scaling about the center
    z_fa=(unif_ell * np.sqrt(Gamma_Threshold)+(z_hat * np.ones((1,m_FA))))

    return np.array(z_fa)

def getellipsoid(crd: np.ndarray,
                 whgts: tp.Optional[tp.Union[np.ndarray, None]]=None, 
                 npoints: int=50,
                 scale: float=3) -> np.ndarray:
    """Randomly and uniformly sample points from the ellipsoid enclosing the molecular system
    The molecular system is scaled by a factor to sample the surrounding space.
    """
    if whgts is not None:
        whgts = np.ones(crd.shape[0])
    print(npoints)
    geom_cntr = np.average(np.array(crd), axis=0)
    dots = sample(np.cov(np.array(crd).T*scale, aweights=whgts),
                  geom_cntr, m_FA=npoints)
    return dots.T

