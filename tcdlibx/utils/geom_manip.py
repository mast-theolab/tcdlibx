import numpy as np

def centre_of_mass(crd: np.ndarray,
    atmass: np.ndarray | list[float]) -> np.ndarray:
    """
    Returns the centre of mass of the coordinates
    
    Arguments:
        crd {[np.array(N,3)]} -- Atomic coordinates
        atmass {[np.array(N)]} -- Atomic masses
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")
    # center of mass
    return np.average(crd, axis=0, weights=atmass)

def inertiatens(crd: np.ndarray,
    atmass: np.ndarray | list[float]) -> np.ndarray:
    """
    Computes and returns the inertia tensor.

    Arguments:
        crd {np.ndarray (N, 3)} -- Atomic coordinates
        atmass {np.ndarray or list of length N} -- Atomic masses

    Returns:
        np.ndarray (3, 3) -- Inertia tensor
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")

    ine_tensor = np.zeros((3, 3), dtype=crd.dtype)
    # inertia tensor
    ine_tensor[0, 0] = (atmass * (crd[:, 1]**2 + crd[:, 2]**2)).sum()
    ine_tensor[1, 1] = (atmass * (crd[:, 0]**2 + crd[:, 2]**2)).sum()
    ine_tensor[2, 2] = (atmass * (crd[:, 0]**2 + crd[:, 1]**2)).sum()
    ine_tensor[0, 1] = ine_tensor[1, 0] = (-atmass * (crd[:, 0] * crd[:, 1])).sum() 
    ine_tensor[0, 2] = ine_tensor[2, 0] = (-atmass * (crd[:, 0] * crd[:, 2])).sum()
    ine_tensor[1, 2] = ine_tensor[2, 1] = (-atmass * (crd[:, 1] * crd[:, 2])).sum()
    return ine_tensor

def inertia(crd: np.ndarray,
    atmass: np.ndarray | list[float]) -> np.ndarray:
    """
    Computes and returns the inertia axis and the rotation values

    Arguments:
        crd {[type]} -- Atomic coordinates
        atmass {[type]} -- Atomic masses
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")
    ine_tensor = inertiatens(crd, atmass)
    # with Upper or not specified is not working -> right-handed
    # eigval, eigvec = np.linalg.eigh(ine_tensor, UPLO='L')
    eigval, eigvec = np.linalg.eigh(ine_tensor)
    return (eigval, eigvec)

def rotmat_principal(crd: np.ndarray,
    atmass: np.ndarray | list[float]) -> np.ndarray:
    """
    Computes and returns the rotation matrix to the principal axis orientation.

    Arguments:
        crd {np.ndarray (N, 3)} -- Atomic coordinates
        atmass {np.ndarray or list of length N} -- Atomic masses
    Raises:
        ValueError: If crd is not a 2D array with shape (N, 3)
        ValueError: If atmass is not a 1D array or list of length N (number of atoms)

    Returns:
        np.ndarray (3, 3) -- Rotation matrix to principal axis orientation
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")
    # compute the centre of mass
    cntrmass = centre_of_mass(crd, atmass)
    # translate the system at the centre of mass
    new_crd = crd - cntrmass
    # compute the inertia axis and rotation matrix
    eigval, rotmat = inertia(new_crd, atmass)

    if np.dot(np.cross(rotmat[:, 0], rotmat[:, 1]), rotmat[:, 2]) < 0:
        #rotmat = rotmat[:, [0, 2, 1]]
        rotmat[:, 0] *= -1
    return rotmat

def eckart_orientation(crd: np.ndarray,
    atmass: np.ndarray | list[float]) -> np.ndarray:
    """
    Returns the coordinates in Eckart orientation
    
    Arguments:
        crd {[np.array(N,3)]} -- Atomic coordinates
        atmass {[np.array(N)]} -- Atomic masses
    Raises:
        ValueError -- If crd is not a 2D array with shape (N, 3)
        ValueError -- If atmass is not a 1D array or list of length N (number of atoms)

    Returns:
        np.ndarray (N, 3) -- Rotation matrix to principal axis orientation
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")
    # compute the centre of mass
    cntrmass = centre_of_mass(crd, atmass)
    # translate the system at the centre of mass
    new_crd = crd - cntrmass
    # compute the inertia axis and rotation matrix
    _, rotmat = inertia(new_crd, atmass)
    # Checks for stereo -> righthanded
    if np.dot(np.cross(rotmat[:, 0], rotmat[:, 1]), rotmat[:, 2]) < 0:
        rotmat[:, 0] *= -1
    # rotate the crd 
    return new_crd @ rotmat # matmul

def traslroto(
    crd: np.ndarray,
    atmass: np.ndarray | list[float]
) -> np.ndarray:
    """
    Computes and returns the translation and rotation vectors (translational and rotational modes)
    for a set of atomic coordinates.

    Arguments:
        crd {np.ndarray (N, 3)} -- Atomic coordinates
        atmass {np.ndarray or list of length N} -- Atomic masses

    Raises:
        ValueError: If crd is not a 2D array with shape (N, 3)
        ValueError: If atmass is not a 1D array or list of length N (number of atoms)
        ValueError: If eigenvalues are too small for a non-single atom system

    Returns:
        np.ndarray (n_tr_rot, 3*N) -- Array of translation and rotation vectors, flattened per atom
    """
    crd = np.asarray(crd)
    atmass = np.asarray(atmass)

    if crd.ndim != 2 or crd.shape[1] != 3:
        raise ValueError("crd must be a 2D numpy array with shape (N, 3)")
    if atmass.ndim != 1 or atmass.shape[0] != crd.shape[0]:
        raise ValueError("atmass must be a 1D array or list of length N (number of atoms)")

    natoms = crd.shape[0]
    # compute the centre of mass
    cntrmass = centre_of_mass(crd, atmass)
    # translate the system at the centre of mass
    new_crd = crd - cntrmass
    # compute the inertia axis and rotation matrix
    eigval, eigvec = inertia(new_crd, atmass)
    new_crd = new_crd @ eigvec

    if not eigval[eigval > 1e-6].shape[0] and natoms != 1:
        raise ValueError('eigval too small')

    lvec_trarot = np.zeros((6, natoms, 3))
    lvec_trarot[0, :, :] = np.sqrt(atmass)[:, np.newaxis] * eigvec[:, 0][np.newaxis, :]
    lvec_trarot[1, :, :] = np.sqrt(atmass)[:, np.newaxis] * eigvec[:, 1][np.newaxis, :]
    lvec_trarot[2, :, :] = np.sqrt(atmass)[:, np.newaxis] * eigvec[:, 2][np.newaxis, :]
    lvec_trarot[3, :, :] = (new_crd[:, 1][:, np.newaxis] * eigvec[:, 2][np.newaxis, :] -
                            new_crd[:, 2][:, np.newaxis] * eigvec[:, 1][np.newaxis, :]) * np.sqrt(atmass)[:, np.newaxis]
    lvec_trarot[4, :, :] = (new_crd[:, 2][:, np.newaxis] * eigvec[:, 0][np.newaxis, :] -
                            new_crd[:, 0][:, np.newaxis] * eigvec[:, 2][np.newaxis, :]) * np.sqrt(atmass)[:, np.newaxis]
    lvec_trarot[5, :, :] = (new_crd[:, 0][:, np.newaxis] * eigvec[:, 1][np.newaxis, :] -
                            new_crd[:, 1][:, np.newaxis] * eigvec[:, 0][np.newaxis, :]) * np.sqrt(atmass)[:, np.newaxis]

    # norm of the tensor
    norm2 = np.einsum('ijk,ijk->i', lvec_trarot, lvec_trarot)
    trot_valid = (norm2 > 1e-8)
    if natoms != 1 and trot_valid.sum() < 6:
        if trot_valid.sum() == 5:
            print('linear?')
        else:
            raise ValueError('eigval too small')
    lvec_trarot = lvec_trarot[trot_valid] / norm2[trot_valid][:, np.newaxis, np.newaxis]

    return lvec_trarot.reshape(-1, 3 * natoms)


def quaternion_rotationmatrix(crd_one, crd_two, weight):
    """[summary]
    See https://doi.org/10.1016/1049-9660(91)90036-O
    TODO add checks on the input data
    Arguments:
        crd_one {[type]} -- the reference Cartesian coordinates
        crd_two {[type]} -- the second cartesian coordinates to be superimpose
        weight {[type]} -- coordinate weight, usually the mass
    """

    def made_mat(rvec, mtype='Q'):
        """Constructs the Quaternion Matrices used in
        https://doi.org/10.1016/1049-9660(91)90036-O
        algorithm
        Arguments:
            rvec {np.ndarray(3+1)} -- Cartesian coordinate
                                    The fourth value can be omitted and it is
                                    the associated scalar
        Keyword Arguments:
            mtype {str} -- type of matrix to be return Q or W (default: {'Q'})
        """
        if rvec.shape[0] != 4:
            r4 = 0.
        else:
            r4 = rvec[3]
        kmat = np.array([[0, -rvec[2], rvec[1]],
                         [rvec[2], 0, -rvec[0]],
                         [-rvec[1], rvec[0], 0]])
        if mtype == 'W':
            kmat *= -1
        resmat = np.identity(4) * r4
        resmat[:3, :3] += kmat
        resmat[:3, 3] = rvec[:3]
        resmat[3, :3] = -rvec[:3]
        return resmat

    elem = crd_one.shape[0]
    qmat = np.zeros((elem, 4, 4))
    wmat = np.zeros_like(qmat)
    for i in range(elem):
        qmat[i, :, :] = made_mat(crd_two[i, :], mtype='Q')
        wmat[i, :, :] = made_mat(crd_one[i, :], mtype='W')
        # 
    qtdotw = np.einsum('mij,mik->mjk', qmat, wmat)
    qtdotw *= weight[:, np.newaxis, np.newaxis]
    amat = qtdotw.sum(axis=0)
    aeval, aevec = np.linalg.eigh(amat)
    del qmat, wmat, amat
    rvec_new = aevec[:, aeval.argmax()]
    # Rotation Matrix
    # [[R, 0], [0, 1]] = W.T dot Q
    qmat_r = made_mat(rvec_new, mtype='Q')
    wmat_r = made_mat(rvec_new, mtype='W')
    rmat = np.einsum('ij,ik->jk', wmat_r, qmat_r)

    return rmat[:3, :3]

def getrotmat(refgeom: np.ndarray,
              newgeom: np.ndarray,
              weights: np.ndarray | None = None) -> np.ndarray:
    """Superimpose the newgeom structure to a reference one (refgeom) using quaternion algorithm.
    Accepts np.ndarray where N is the number of atoms
                
    Arguments:
        refgeom {np.ndarray(N, 3)} -- Reference structure 
        newgeom {np.ndarray(N, 3)} -- Structure to be superimposed 
    
    Keyword Arguments:
        weights {np.ndarray(N) | None} -- coordinate weight, usually the mass (default: {None})
    
    Returns:
        np.ndarray(3, 3) -- Rotation Matrix 
    """

    if weights is None:
        weights = np.array(1)
    cm_ref = centre_of_mass(refgeom, weights)
    cm_new = centre_of_mass(newgeom, weights)
    cm_refgeom = refgeom - cm_ref
    cm_newgeom = newgeom - cm_new
    rotmat = quaternion_v3(cm_refgeom, cm_newgeom, weights)
    return rotmat

def quaternion_v3(crd_one, crd_two, weights):
    """[summary]
    # Adapted from geomband
    Kneller, Molecular Simulation, 1991, Vol. 7, pp. 113-119
    TODO add checks on the input data
    Arguments:
        crd_one {[type]} -- The reference cartesian coordinates
        crd_two {[type]} -- The cartesian coordinates to be superimposed
        weights {[type]} -- Weights of the coordinates (Es. Atomic mass)
    """
    # https://arxiv.org/pdf/physics/0506177.pdf
    # E = 1/W sum_k(w_k |y_k - T(x_k)|**2)
    # W = sum_k(w_k)
    # wtot = weights.sum()
    # T(x) =Rq(x) +d
    elem = crd_one.shape[0]
    mmat = np.zeros((elem, 4, 4))
    def norm(x):
        return np.dot(x, x)
    for i in range(elem):
        tmp_sq = norm(crd_one[i, :]) + norm(crd_two[i, :])
        mmat[i, 0, 0] = tmp_sq - 2 * (crd_one[i, :]*crd_two[i, :]).sum()
        mmat[i, 0, 1] = 2 * (crd_one[i, 2]*crd_two[i, 1] - crd_one[i, 1]*crd_two[i, 2])
        mmat[i, 0, 2] = 2 * (-crd_one[i, 2]*crd_two[i, 0] + crd_one[i, 0]* crd_two[i, 2])
        mmat[i, 0, 3] = 2 * (crd_one[i, 1]*crd_two[i, 0] - crd_one[i, 0]*crd_two[i, 1])
        mmat[i, 1, 1] = tmp_sq + 2 * (crd_one[i]*crd_two[i]*[-1, 1, 1]).sum()
        mmat[i, 1, 2] = -2 * (crd_one[i, 1]*crd_two[i, 0] + crd_one[i, 0]*crd_two[i, 1])
        mmat[i, 1, 3] = -2 * (crd_one[i, 2]*crd_two[i, 0] + crd_one[i, 0]*crd_two[i, 2])
        mmat[i, 2, 2] = tmp_sq + 2 * (crd_one[i]*crd_two[i]*[1, -1, 1]).sum()
        mmat[i, 2, 3] = -2 * (crd_one[i, 2]*crd_two[i, 1] + crd_one[i, 1]*crd_two[i, 2])
        mmat[i, 3, 3] = tmp_sq + 2 * (crd_one[i]*crd_two[i]*[1, 1, -1]).sum()
        mmat[i, :, :] *= weights[i]
    mmat = mmat.sum(axis=0)
    meval, mevec = np.linalg.eigh(mmat, UPLO='U')

    rvec = mevec[:, meval.argmin()]
    rmat = np.array([[rvec[0]**2 + rvec[1]**2 - rvec[2]**2 - rvec[3]**2,
                      2*(-rvec[0]*rvec[3]+rvec[1]*rvec[2]),
                      2*(rvec[0]*rvec[2]+rvec[1]*rvec[3])],
                     [2*(rvec[0]*rvec[3]+rvec[1]*rvec[2]),
                      rvec[0]**2 - rvec[1]**2 + rvec[2]**2 - rvec[3]**2,
                      2*(-rvec[0]*rvec[1]+rvec[2]*rvec[3])],
                     [2*(-rvec[0]*rvec[2]+rvec[1]*rvec[3]),
                      2*(rvec[0]*rvec[1]+rvec[2]*rvec[3]),
                      rvec[0]**2 - rvec[1]**2 - rvec[2]**2 + rvec[3]**2]])
    return rmat

def superimpose(refgeom: np.ndarray,
                newgeom: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Superimpose the newgeom structure to a reference one (refgeom) using quaternion algorithm.
    Accepts np.ndarray where N is the number of atoms
                
    Arguments:
        refgeom {np.ndarray(N, 3)} -- Reference structure 
        newgeom {np.ndarray(N, 3)} -- Structure to be superimposed 
    
    Keyword Arguments:
        weights {np.ndarray(N) | None} -- coordinate weight, usually the mass (default: {None})
    
    Returns:
        np.ndarray(N, 3) -- The transformed structure 
    """

    if weights is None:
        weights = np.array(1)
    cm_ref = centre_of_mass(refgeom, weights)
    cm_new = centre_of_mass(newgeom, weights)
    cm_refgeom = refgeom - cm_ref
    cm_newgeom = newgeom - cm_new
    rotmat = quaternion_v3(cm_refgeom, cm_newgeom, weights)
    #return (cm_newgeom - cm_new + cm_ref) @ rotmat
    return (cm_newgeom@rotmat) + cm_ref

def superimpose_rotmat(refgeom: np.ndarray,
                newgeom: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Superimpose the newgeom structure to a reference one (refgeom) using quaternion algorithm.
    Accepts np.ndarray where N is the number of atoms
                
    Arguments:
        refgeom {np.ndarray(N, 3)} -- Reference structure 
        newgeom {np.ndarray(N, 3)} -- Structure to be superimposed 
    
    Keyword Arguments:
        weights {np.ndarray(N) | None} -- coordinate weight, usually the mass (default: {None})
    
    Returns:
        np.ndarray(N, 3) -- The transformed structure 
    """

    if weights is None:
        weights = np.array(1)
    cm_ref = centre_of_mass(refgeom, weights)
    cm_new = centre_of_mass(newgeom, weights)
    cm_refgeom = refgeom - cm_ref
    cm_newgeom = newgeom - cm_new
    rotmat = quaternion_v3(cm_refgeom, cm_newgeom, weights)
    return rotmat


