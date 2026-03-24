#!/bin/python3
"""
Libs to plot the vector fields cube files
"""
import sys
import operator
import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as a3d

from estampes.data.physics import PHYSFACT

from tcdlibx.utils.custom_except import NoValidData
from tcdlibx.utils.mol_data import ELEMENTS, AT_COL2, AT_RAD, AT_RVDW
from tcdlibx.utils.var_tools import Streamlines


def check_ax(axis):
    """
    Check the axis Number and assign the other two
    """
    if axis == 'x':
        it0 = 0
    elif axis == 'y':
        it0 = 1
    elif axis == 'z':
        it0 = 2
    else:
        try:
            it0 = int(axis)
            if it0 < 0 or it0 > 2:
                print('Wrong axis')
                raise NoValidData(it0, "not a valid axis")
        except ValueError:
            print('Could not guess the type of axis')
            sys.exit()
        except NoValidData as err:
            print("{}:{}".format(err.expression, err.message))
            sys.exit()

    i_one = 1 >> it0
    i_two = 2 - (1 >> (2 - it0))

    return (it0, i_one, i_two)


def calc_off(axis, in_0, in_1, in_2, npts):
    """
    Calulate the off-set for the reduction
    """
    if not axis:
        ioff = in_1 * npts[2] + in_2 + in_0 * npts[2] * npts[1]
    elif not axis - 1:
        ioff = in_1 * npts[2] * npts[1] + in_2 + in_0 * npts[2]
    else:
        ioff = in_1 * npts[2] * npts[1] + in_2 * npts[2] + in_0

    return ioff


def set_subgrid(cubdata, xmin=None, xmax=None,
                ymin=None, ymax=None,
                zmin=None, zmax=None):
    """Build subgrid of vectors within box"""
    tmp_cube = np.copy(cubdata.cube)
    i123 = 0
    for ith1 in range(cubdata.npts[0]):
        x_pnt = cubdata.box[0, ith1*cubdata.npts[1]*cubdata.npts[2]]
        cleanx = (xmin is not None and x_pnt < xmin) or \
                 (xmax is not None and x_pnt > xmax)
        for ith2 in range(cubdata.npts[1]):
            y_pnt = cubdata.box[1, ith2*cubdata.npts[2]]
            cleany = (ymin is not None and y_pnt < ymin) or \
                     (ymax is not None and y_pnt > ymax)
            for ith3 in range(cubdata.npts[2]):
                z_pnt = cubdata.box[2, ith3]
                cleanz = (zmin is not None and z_pnt < zmin) or \
                         (zmax is not None and z_pnt > zmax)
                if cleanx or cleany or cleanz:
                    tmp_cube[:, i123] = 0.0
                i123 += 1
    return tmp_cube


def to_grid(vec, size):
    """
    helper function to convert the vector data to a grid
    """
    tosize = (vec.shape[0],) + (size[1], size[0])
    grid = np.zeros(tosize)
    for i in range(tosize[0]):
        grid[i, :, :] = vec[i].reshape(size).T

    return grid


def simp_proj(cubdat1, axis, grid=False):
    """
    Project The cube data set on 2D plane
    orthogonal to the selected axis
    """

    ithx, ith_one, ith_two = check_ax(axis)
    # print(ithx, ith_one, ith_two)

    box_tmp = np.zeros((2, cubdat1.npts[ith_one] * cubdat1.npts[ith_two]))
    vec_tmp = np.zeros_like((box_tmp))
    it_123 = 0
    for ith_2 in range(cubdat1.npts[ith_one]):
        for ith_3 in range(cubdat1.npts[ith_two]):
            for ith_1 in range(cubdat1.npts[ithx]):
                i123 = calc_off(ithx, ith_1, ith_2, ith_3, cubdat1.npts)
                vec_tmp[0, it_123] += cubdat1.cube[ith_one, i123]
                vec_tmp[1, it_123] += cubdat1.cube[ith_two, i123]
            ioff = calc_off(ithx, 0, ith_2, ith_3, cubdat1.npts)
            box_tmp[0, it_123] = cubdat1.box[ith_one, ioff]
            box_tmp[1, it_123] = cubdat1.box[ith_two, ioff]
            it_123 += 1
    if not grid:
        return (vec_tmp, box_tmp)
    ax_dim = (cubdat1.npts[ith_one], cubdat1.npts[ith_two])
    return (to_grid(vec_tmp, ax_dim), to_grid(box_tmp, ax_dim))


def draw_mol2d(ax_out, coord, atnum, axis, scale,
               conmat=None, to_bohr=False, vollimit=None):
    """
    Draw the molecule in 2D as balls and sticks
    """
    # rscl = 80
    rscl = 160
    if to_bohr:
        cscl = 1.
    else:
        cscl = 1.*PHYSFACT.bohr2ang
    ithx, ith_one, ith_two = check_ax(axis)

    inside_atom = np.ones(len(coord), dtype=bool)
    if vollimit is not None:
        if len(vollimit) != 6:
            raise ValueError(
                "vollimit must contain [xmin, xmax, ymin, ymax, zmin, zmax]"
            )
        xmin, xmax, ymin, ymax, zmin, zmax = vollimit
        for i_a, crd in enumerate(coord):
            inside_atom[i_a] = (
                (xmin is None or crd[0] >= xmin) and
                (xmax is None or crd[0] <= xmax) and
                (ymin is None or crd[1] >= ymin) and
                (ymax is None or crd[1] <= ymax) and
                (zmin is None or crd[2] >= zmin) and
                (zmax is None or crd[2] <= zmax)
            )

    if conmat:
        for bond in conmat:
            i_a, j_a = bond
            x_crd = [coord[i_a-1][ith_one]*cscl, coord[j_a-1][ith_one]*cscl]
            y_crd = [coord[i_a-1][ith_two]*cscl, coord[j_a-1][ith_two]*cscl]
            balpha = 1.0
            if not inside_atom[i_a-1] or not inside_atom[j_a-1]:
                balpha = 0.3
            ax_out.plot(x_crd, y_crd, c='#555555', lw=2.5 / scale[1],
                        zorder=1, alpha=balpha)
    # c_sorted = sorted(ax_val.items(), key=operator.itemgetter(1), reverse=True)
    ax_val = {}
    for i_a, crd in enumerate(coord):
        ax_val[i_a] = crd[ithx]
    c_sorted = sorted(ax_val.items(), key=operator.itemgetter(1))
    x_crd, y_crd, col, edge_col, rad = [], [], [], [], []
    for i_a, xval in c_sorted:
        del xval
        x_crd.append(coord[i_a][ith_one]*cscl)
        y_crd.append(coord[i_a][ith_two]*cscl)
        ian_l = int(atnum[i_a])
        rad.append(AT_RAD[ian_l]*rscl/scale[0])
        alpha = 1.0 if inside_atom[i_a] else 0.3
        at_col = tuple(AT_COL2[ian_l])
        col.append((at_col[0], at_col[1], at_col[2], alpha))
        edge_col.append((53/255, 53/255, 53/255, alpha))
    ax_out.scatter(x_crd, y_crd, s=rad, c=col, edgecolors=edge_col, zorder=2)


def draw_nm2d(ax_out, coord, e_vec, axis, scal_f, to_bohr=False):
    """
    Draw the Normal Modes of a vibration
    in the 2D plot
    """
    if to_bohr:
        cscl = 1.
    else:
        cscl = 1.*PHYSFACT.bohr2ang

    ithx, ith_one, ith_two = check_ax(axis)
    del ithx

    atom_num = len(coord)
    xth, yth, uth, vth = [], [], [], []
    for i in range(atom_num):
        xth.append(coord[i][ith_one]*cscl)
        yth.append(coord[i][ith_two]*cscl)
        uth.append(e_vec[i * 3 + ith_one]*1/scal_f)
        vth.append(e_vec[i * 3 + ith_two]*1/scal_f)
    ax_out.quiver(xth, yth, uth, vth, units='width',
                  zorder=1, headwidth=2, headlength=4,
                  headaxislength=3.5, scale=3, color='b')


def draw_nm2dcw(ax_out, coord, e_vec,
                atnums, axis, scal_f, to_bohr=False):
    """
    Draw the Normal Modes of a vibration
    in the 2D plot
    """
    if to_bohr:
        cscl = 1.
    else:
        cscl = 1.*PHYSFACT.bohr2ang

    ithx, ith_one, ith_two = check_ax(axis)
    del ithx

    atom_num = len(coord)
    xth, yth, uth, vth = [], [], [], []
    for i in range(atom_num):
        xth.append(coord[i][ith_one]*cscl)
        yth.append(coord[i][ith_two]*cscl)
        uth.append(e_vec[i * 3 + ith_one]*(-atnums[i])/scal_f)
        vth.append(e_vec[i * 3 + ith_two]*(-atnums[i])/scal_f)
    ax_out.quiver(xth, yth, uth, vth, units='width',
                  zorder=1, headwidth=2, headlength=4,
                  headaxislength=3.5, scale=3, color='r')


def draw_cube(ax1, cbdat, lvec=0.1):
    """
    Plots the cubedataset as 3Dquiver in ax
    lvec: the scaling factor
    """
    # Remember to plot!!!
    # fig = plt.figure()
    # fig.clf()
    # ax = fig.gca(projection='3d')
    colma = cm.get_cmap("seismic")
    ax1.quiver(cbdat.box[0, :], cbdat.box[1, :],
               cbdat.box[2, :], cbdat.cube[0, :],
               cbdat.cube[1, :], cbdat.cube[2, :],
               length=lvec, cmap=colma, normalize=False)


def draw_mol(ax1, coord, atnum, conmat=None, to_bohr=False):
    """
    Add the molecule in 3D as scattered point
    ax1: ax where to put the graphic
    coord: atomic coordinates
    atnum: atomic numbers
    conmat: connectivity matrix as dictionary
    to_bohr: the lenght unit
    """
    atcrd = {}  # coordinates per atom type
    rscl = 80
    if to_bohr:
        cscl = 1.
    else:
        cscl = 1.*PHYSFACT.bohr2ang
    for iat, iatn in enumerate(atnum):
        if iatn not in atcrd:
            atcrd[iatn] = []
        atcrd[iatn].append(coord[iat])
    for iatn in atcrd:
        print(iatn)
        x_cr, y_cr, z_cr = [], [], []
        for itc in atcrd[iatn]:
            x_cr.append(itc[0]*cscl)
            y_cr.append(itc[1]*cscl)
            z_cr.append(itc[2]*cscl)
        print(AT_RAD[int(iatn)]*rscl)
        ax1.scatter(x_cr, y_cr, z_cr, s=AT_RAD[int(iatn)]*rscl,
                    c=AT_COL2[int(iatn)], label=ELEMENTS[int(iatn)],
                    depthshade=True)
    if conmat:
        for bond in conmat:
            iat, jat = bond
            x_cr = [coord[iat-1][0]*cscl, coord[jat-1][0]*cscl]
            y_cr = [coord[iat-1][1]*cscl, coord[jat-1][1]*cscl]
            z_cr = [coord[iat-1][2]*cscl, coord[jat-1][2]*cscl]
            line = a3d.Line3D(x_cr, y_cr, z_cr, c='#555555', lw=2.5)
            ax1.add_line(line)


def stream_plt(ax0, cubdat, axis):
    """
    Function for plot a stream plot starting from
    """
    vec2, box2 = simp_proj(cubdat, axis)
    # if sym == 'x':
    #     half = 0
    # elif sym == 'y':
    #     half = 1
    ithx, ith_one, ith_two = check_ax(axis)
    del ithx

    # ax_dim = (cubdat.npts[ith_two], cubdat.npts[ith_one])
    ax_dim = (cubdat.npts[ith_one], cubdat.npts[ith_two])
    # To Grid!
    # starts = np.copy(box2).T
    vec2 = to_grid(vec2, ax_dim)
    box2 = to_grid(box2, ax_dim)
    mod = np.sqrt(vec2[0, :, :]*vec2[0, :, :] +
                  vec2[1, :, :]*vec2[1, :, :])
    liwg = 3 * mod  / mod.max()
    mod = np.log(mod)
    # if sym:
    #     mask = box2[half, :, :] > 0
    #     vecmask_1 = vec2.copy()
    #     vecmask_2 = vec2.copy()
    #     vecmask_1[1, mask] = None
    #     vecmask_2[1, ~mask] = None
    #     strm = ax0.streamplot(box2[0, :, :], box2[1, :, :],
    #                           vecmask_1[0, :, :], vecmask_1[1, :, :],
    #                           density=[2.7, 2.7],
    #                           start_points=starts,
    #                           arrowstyle='Fancy',
    #                           color=mod, linewidth=liwg,
    #                           cmap=cm.PuRd)
    # liwg = to_grid(liwg, ax_dim)
    # mod = to_grid(mod, ax_dim)
    colma = cm.get_cmap("Blues")
    strm = ax0.streamplot(box2[0, :, :], box2[1, :, :],
                          vec2[0, :, :], vec2[1, :, :],
                          density=[2.7, 2.7],
                          # start_points=starts,
                          arrowstyle='Fancy',
                          color=mod, linewidth=liwg,
                          cmap=colma)
                        # cmap=cm.PuRd)
    # Just a reminder: fig0.colorbar(strm.lines)

    return strm


def stream2_plt(ax0, cubdat, axis, loops=False, background=True):
    """
    Function for plot a stream plot starting from
    """
    vec2, box2 = simp_proj(cubdat, axis)
    ithx, ith_one, ith_two = check_ax(axis)
    del ithx
    # ax_dim = (cubdat.npts[ith_two], cubdat.npts[ith_one])
    ax_dim = (cubdat.npts[ith_one], cubdat.npts[ith_two])
    # To Grid!
    norm = (np.linalg.norm(vec2, axis=0)).reshape(ax_dim).T
    vec2 = to_grid(vec2, ax_dim) * -1  # BUG I'm still not sure why
    box2 = to_grid(box2, ax_dim)
    # Filtering a bit
    # lwth = norm.max() / 10000  # try it
    # vec2[0, np.where(norm < lwth)] = 0
    # vec2[1, np.where(norm < lwth)] = 0

    lengths = []
    colors = []
    lines = []
    # The stream object
    stm = Streamlines(box2[0, :, :], box2[1, :, :],
                      vec2[0, :, :], vec2[1, :, :],
                      spacing=4,
                      detectLoops=loops)
    for streamline in stm.streamlines:
        xline, yline = streamline
        points = np.array([xline, yline]).T.reshape(-1, 1, 2)
        filrm = []
        for i in range(len(xline)-1):
            cop = [xline[i], yline[i]]
            if cop[0] > box2[0, :, :].max():
                cop[0] = box2[0, :, :].max()
            elif cop[0] < box2[0, :, :].min():
                cop[0] = box2[0, :, :].min()
            if cop[1] > box2[1, :, :].max():
                cop[1] = box2[1, :, :].max()
            elif cop[1] < box2[1, :, :].min():
                cop[1] = box2[1, :, :].min()
            try:
                tmp = stm._interp(cop[0], cop[1])
            except IndexError:
                # Not very elegant
                tmp = [0, 0.]
            filrm.append(np.sqrt(tmp[0]**2+tmp[1]**2))
        filrm = np.array(filrm)
        filrm = np.mean(np.concatenate([[filrm[:-1], filrm[1:]]], axis=1),
                        axis=0)
        # Filtering the streamlines besed on the field
        # by reducing the linewidth
        filrm *= 0.5/norm.max()
        # index1 = np.where(filrm < lwth * 500)
        # index2 = np.where(filrm < lwth * 100)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        num = len(segments)
        dis = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=-1))
        lun = dis.cumsum().reshape(num, 1) + np.random.uniform(0, 1)
        del dis
        col = np.zeros((num, 3))
        col[:] = (lun*1.5) % 1
        linewidths = filrm
        # linewidths = np.zeros(num)
        # linewidths[:] = 1.5 - ((L.reshape(n)*1.5) % 1)
        # Default widith
        # linewidths[:] = 0.5
        # linewidths[:] = 0.5
        # Scaled
        # linewidths[index1] = 0.05
        # linewidths[index2] = 0.005
        # print(linewidths)
        line = LineCollection(segments, color=col, linewidth=linewidths)
        # line = LineCollection(segments, color=C, linewidth=0.5)
        lengths.append(lun)
        colors.append(col)
        lines.append(line)
        ax0.add_collection(line)
    # Add the background representing the field magnitude
    if background:
        ax0.imshow(norm, extent=[box2[0, :, :].min(),
                                 box2[0, :, :].max(),
                                 box2[1, :, :].max(),
                                 box2[1, :, :].min()],
                   interpolation='bilinear', cmap='Blues')

    return(lengths, colors, lines)

def animated_stream(fig0, params, fname='test.gif', nfr=27, savegif=False):
    """
    Animates the streamlige obtained with stream2_plt
    """
    lengths, colors, lines = params
    def update(frame_no):
        """
        Update function for animation
        """
        for i in range(len(lines)):
            lengths[i] += 0.05
            colors[i][:] = (lengths[i]*1.5) % 1
            lines[i].set_color(colors[i])
    animation = FuncAnimation(fig0, update, frames=nfr, interval=20)
    if savegif:
        animation.save(fname, writer='imagemagick', fps=30)

    return animation


def quiver_plt(ax0, cubdat, axis, vscale,
               filtred=False, col='black', alf=1, background=False):
    """
    TODO description
    """
    vec2, box2 = simp_proj(cubdat, axis)
    ithx, ith_one, ith_two = check_ax(axis)
    ax_dim = (cubdat.npts[ith_one], cubdat.npts[ith_two])
    if filtred:
        vec_norm = np.sqrt((vec2 ** 2).sum(axis=0))
        # index = np.where(vec_norm > 5e-2)
        vec_max = vec_norm.max()
        index2 = np.where(vec_norm > vec_max * 0.5)
        vec2[:, index2] *= 0.5
        # box2 = box2[:, index]
        # vec2 = vec2[:, index]
    quiv = ax0.quiver(box2[0, :], box2[1, :], vec2[0, :],
                      vec2[1, :], units='width',
                      scale=vscale, color=col, alpha=alf,
                      zorder=3)
    if background:
        vec_norm = (np.linalg.norm(vec2, axis=0)).reshape(ax_dim).T
        ax0.imshow(vec_norm, extent=[box2[0, :].min(),
                                 box2[0, :].max(),
                                 box2[1, :].max(),
                                 box2[1, :].min()],
                   interpolation='bilinear', cmap='Blues')
    return quiv


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
