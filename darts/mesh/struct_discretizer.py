# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
import time
from .transcalc import *

# Class definition of the StructDiscretizer, which discretizes a structured reservoir
class StructDiscretizer:
    # Define Darcy constant for changes to correct units:
    darcy_constant = 0.0085267146719160104986876640419948

    def __init__(self, nx, ny, nz, dx, dy, dz, permx, permy, permz, global_to_local = 0, coord = 0, zcorn = 0,
                 is_cpg = False):
        """
        Class constructor method
        :param nx: number of reservoir blocks in the x-direction
        :param ny: number of reservoir blocks in the y-direction
        :param nz: number of reservoir blocks in the z-direction
        :param dx: size of the reservoir blocks in the x-direction (scalar or vector form) [m]
        :param dy: size of the reservoir blocks in the y-direction (scalar or vector form) [m]
        :param dz: size of the reservoir blocks in the z-direction (scalar or vector form) [m]
        :param permx: permeability of the reservoir blocks in the x-direction (scalar or vector form) [mD]
        :param permy: permeability of the reservoir blocks in the y-direction (scalar or vector form) [mD]
        :param permz: permeability of the reservoir blocks in the z-direction (scalar or vector form) [mD]
        :param global_to_local: one can define arbitrary indexing (mapping from global to local) for local
                                arrays. Default indexing is by X (fastest),then Y, and finally Z (slowest)
        """
        # For users new to Object Oriented Programming, please see:
        #   - https: // www.programiz.com / python - programming / object - oriented - programming
        #   - https: // www.tutorialspoint.com / python / python_classes_objects.htm
        #   - https: // python.swaroopch.com / oop.html
        # The "self." refers to the object of the class StructDiscretizer. So after constructing an instance of the
        # class: instance_of_discr_class =  StructDiscretizer(nx, ny, nz, dx, dy, dz, permx, permy, permz),
        # the variables, e.g. nx, can be accessed as instance_of_discr_class.nx (hence the self.nx). For more
        # information please see the links above!
        # Store object variables, part of the discretization information:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nodes_tot = nx * ny * nz
        self.arr_shape = (nx, ny, nz)
        self.min_tran_tranD = 1e-5
        self.eps_div = 1e-8


        self.is_cpg = is_cpg
        if self.is_cpg:
            print("Calculating CPG grid...", end='', flush=True)
            #self.vectorized_cpg(coord, zcorn)
            plain_points_num = (nx + 1) * (ny + 1)
            cells_num = nx * ny * nz
            assert (zcorn.size == 8 * cells_num)
            assert (coord.size == 6 * plain_points_num)
            zcorn = zcorn.reshape((2*nz, 2*ny, 2*nx))
            dtype = [('center', np.float64, (3,)), ('faces', np.float64, (6,2,3)), ('area', np.float64, (6,))]
            self.volume = np.zeros(shape=self.arr_shape)
            self.cell_data = np.empty(shape=self.arr_shape, dtype=dtype)
            for j in range(ny):
                for i in range(nx):
                    xm_ym = i + j * (nx + 1)
                    xp_ym = i + 1 + j * (nx + 1)
                    xm_yp = i + (j + 1) * (nx + 1)
                    xp_yp = i + 1 + (j + 1) * (nx + 1)
                    v2d = np.array([coord[6 * xm_ym:6 * (xm_ym + 1)].reshape(2, 3),
                                    coord[6 * xp_ym:6 * (xp_ym + 1)].reshape(2, 3),
                                    coord[6 * xm_yp:6 * (xm_yp + 1)].reshape(2, 3),
                                    coord[6 * xp_yp:6 * (xp_yp + 1)].reshape(2, 3)])
                    z_top = np.concatenate((zcorn[0, 2*j,2*i:2*i+2], zcorn[0, 2*j+1,2*i:2*i+2]))
                    z_bot = np.concatenate((zcorn[-1, 2*j,2*i:2*i+2], zcorn[-1, 2*j+1,2*i:2*i+2]))
                    zero_ids = np.argwhere(z_bot - z_top == 0)[:, 0]
                    for k in range(nz):
                        id = i + nx * (j + k * ny)
                        z_cur = np.concatenate((zcorn[2*k, 2*j,2*i:2*i+2], zcorn[2*k, 2*j+1,2*i:2*i+2],
                                                zcorn[2*k+1, 2*j,2*i:2*i+2], zcorn[2*k+1, 2*j+1,2*i:2*i+2]))
                        v_top = v2d[:, 0] + (v2d[:, 1] - v2d[:, 0]) * ((z_cur[:4] - z_top)
                                                                       / (self.eps_div + z_bot - z_top))[:,np.newaxis]
                        v_bot = v2d[:, 0] + (v2d[:, 1] - v2d[:, 0]) * ((z_cur[4:] - z_top)
                                                                       / (self.eps_div + z_bot - z_top))[:,np.newaxis]
                        v_top[zero_ids] = v2d[zero_ids, 0]
                        v_bot[zero_ids] = v2d[zero_ids, 1]
                        v = np.concatenate((np.c_[v_top[:, :2], z_cur[:4]], np.c_[v_bot[:, :2], z_cur[4:]]))

                        a1, n1, c1 = self.calc_area_and_centroid(np.array([v[0], v[2], v[6], v[4]]))
                        a2, n2, c2 = self.calc_area_and_centroid(np.array([v[1], v[3], v[7], v[5]]))
                        a3, n3, c3 = self.calc_area_and_centroid(np.array([v[1], v[0], v[4], v[5]]))
                        a4, n4, c4 = self.calc_area_and_centroid(np.array([v[3], v[2], v[6], v[7]]))
                        a5, n5, c5 = self.calc_area_and_centroid(np.array([v[0], v[1], v[3], v[2]]))
                        a6, n6, c6 = self.calc_area_and_centroid(np.array([v[4], v[5], v[7], v[6]]))
                        #s = a1 + a2 + a3 + a4 + a5 + a6
                        #c_faces = (a1*c1 + a2*c2 + a3*c3 + a4*c4 + a5*c5 + a6*c6) / s if s > 0 else c1
                        vol, c = self.calc_hexahedron_volume_and_centroid(v, np.array([n1, n2, n3, n4, n5, n6]), np.array([c1, c2, c3, c4, c5, c6]), np.array([a1, a2, a3, a4, a5, a6]))
                        self.volume[i, j, k] = vol
                        self.cell_data[i, j, k] = (c, np.array([[n1, c1], [n2, c2], [n3, c3], [n4, c4], [n5, c5], [n6, c6]]),
                                                   np.array([a1, a2, a3, a4, a5, a6]))
            self.len_cell_xminus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 0, 1], axis=3)
            self.len_cell_xplus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 1, 1], axis=3)
            self.len_cell_yminus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 2, 1], axis=3)
            self.len_cell_yplus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 3, 1], axis=3)
            self.len_cell_zminus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 4, 1], axis=3)
            self.len_cell_zplus = np.linalg.norm(self.cell_data['center'] - self.cell_data['faces'][:, :, :, 5, 1], axis=3)
            self.dist_cell_x = np.linalg.norm(self.cell_data[1:,:,:]['center'] - self.cell_data[:-1,:,:]['center'], axis=3)
            self.dist_cell_y = np.linalg.norm(self.cell_data[:,1:,:]['center'] - self.cell_data[:,:-1,:]['center'], axis=3)
            self.dist_cell_z = np.linalg.norm(self.cell_data[:,:,1:]['center'] - self.cell_data[:,:,:-1]['center'], axis=3)
            self.perm_x_cell = self.convert_to_3d_array(permx, 'permx')
            self.perm_y_cell = self.convert_to_3d_array(permy, 'permy')
            self.perm_z_cell = self.convert_to_3d_array(permz, 'permz')
            self.perm_xminus =  (self.perm_x_cell * self.cell_data['faces'][:, :, :, 0, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 0, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 0, 0, 2] ** 2)
            self.perm_xplus =   (self.perm_x_cell * self.cell_data['faces'][:, :, :, 1, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 1, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 1, 0, 2] ** 2)
            self.perm_yminus =  (self.perm_x_cell * self.cell_data['faces'][:, :, :, 2, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 2, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 2, 0, 2] ** 2)
            self.perm_yplus =   (self.perm_x_cell * self.cell_data['faces'][:, :, :, 3, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 3, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 3, 0, 2] ** 2)
            self.perm_zminus =  (self.perm_x_cell * self.cell_data['faces'][:, :, :, 4, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 4, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 4, 0, 2] ** 2)
            self.perm_zplus =   (self.perm_x_cell * self.cell_data['faces'][:, :, :, 5, 0, 0] ** 2 + \
                                self.perm_y_cell * self.cell_data['faces'][:, :, :, 5, 0, 1] ** 2 + \
                                self.perm_z_cell * self.cell_data['faces'][:, :, :, 5, 0, 2] ** 2)

            print(" done.")

        # If scalar dx, dy, and dz are specified: Store constant control volume dimensions
        else:
            self.len_cell_xdir = self.convert_to_3d_array(dx, 'dx')
            self.len_cell_ydir = self.convert_to_3d_array(dy, 'dy')
            self.len_cell_zdir = self.convert_to_3d_array(dz, 'dz')
            self.volume = self.len_cell_xdir * self.len_cell_ydir * self.len_cell_zdir
        self.perm_x_cell = self.convert_to_3d_array(permx, 'permx')
        self.perm_y_cell = self.convert_to_3d_array(permy, 'permy')
        self.perm_z_cell = self.convert_to_3d_array(permz, 'permz')

        # Initialize mapping arrays assuming all cells are active
        if np.isscalar(global_to_local):
            # default indexing
            self.global_to_local = np.arange(self.nodes_tot, dtype=np.int32)
            self.local_to_global = np.arange(self.nodes_tot, dtype=np.int32)
        else:
            # external arbitrary indexing
            self.global_to_local = global_to_local
            self.local_to_global = np.zeros(self.nodes_tot, dtype=np.int32)
            self.local_to_global[self.global_to_local] = np.arange(self.nodes_tot, dtype=np.int32)

    def vectorized_cpg(self, coord, zcorn):
        coord = np.reshape(coord,(int(coord.size/6),6))
        zcorn = zcorn.reshape((2 * self.nz, 2 * self.ny, 2 * self.nx))

        i_ids = np.arange(self.nx, dtype=np.intp)
        j_ids = np.arange(self.ny, dtype=np.intp)
        k_ids = np.arange(self.nz, dtype=np.intp)
        i = np.lib.stride_tricks.as_strided(i_ids, (self.nz, self.ny, self.nx), (0, 0, i_ids.strides[0])).flatten()
        j = np.lib.stride_tricks.as_strided(j_ids, (self.nz, self.ny, self.nx), (0, j_ids.strides[0], 0)).flatten()
        k = np.lib.stride_tricks.as_strided(k_ids, (self.nz, self.ny, self.nx), (k_ids.strides[0], 0, 0)).flatten()
        i2d = i[:self.nx*self.ny]
        j2d = j[:self.nx*self.ny]
        #i = np.tile(np.arange(self.nx, dtype=np.intp), (self.ny,1))
        #j = np.tile(np.array([np.arange(self.ny, dtype=np.intp)]).transpose(), (1, self.nx))
        xm_ym = i2d + j2d * (self.nx + 1)
        xp_ym = i2d + 1 + j2d * (self.nx + 1)
        xm_yp = i2d + (j2d + 1) * (self.nx + 1)
        xp_yp = i2d + 1 + (j2d + 1) * (self.nx + 1)
        v2d = np.swapaxes(np.array([coord[xm_ym], coord[xp_ym], coord[xm_yp], coord[xp_yp]]), 0, 1)
        z_top = np.array([zcorn[0, 2*j2d, 2*i2d], zcorn[0, 2*j2d, 2*i2d+1], zcorn[0, 2*j2d+1, 2*i2d], zcorn[0, 2*j2d+1, 2*i2d+1]]).T
        z_bot = np.array([zcorn[-1, 2*j2d, 2*i2d], zcorn[-1, 2*j2d, 2*i2d+1], zcorn[-1, 2*j2d+1, 2*i2d], zcorn[-1, 2*j2d+1, 2*i2d+1]]).T
        zero_ids = np.argwhere(z_bot - z_top == 0)
        z_cur = np.array([zcorn[2 * k, 2 * j, 2 * i],           zcorn[2 * k, 2 * j, 2 * i + 1],
                          zcorn[2 * k, 2 * j + 1, 2 * i],       zcorn[2 * k, 2 * j + 1, 2 * i + 1],
                          zcorn[2 * k + 1, 2 * j, 2 * i],       zcorn[2 * k + 1, 2 * j, 2 * i + 1],
                          zcorn[2 * k + 1, 2 * j + 1, 2 * i],   zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1]]).T

        #v_top = v2d[:, 0] + (v2d[:, 1] - v2d[:, 0]) * ((z_cur[:4] - z_top)
        #                                               / (self.eps_div + z_bot - z_top))[:, np.newaxis]
        #v_bot = v2d[:, 1] + (v2d[:, 1] - v2d[:, 0]) * ((z_cur[4:] - z_top)
        #                                               / (self.eps_div + z_bot - z_top))[:, np.newaxis]

        return 0

    def calc_area_and_centroid(self, pts):
        # point must be in circular order
        n_pts = pts.shape[0]
        #area = 0.0
        #c = np.zeros(3)
        #n = np.zeros(3)
        n_all = np.cross(pts[1:-1] - pts[0], pts[2:] - pts[0]) / 2.0
        n = np.sum(n_all, axis=0)
        area_all = np.linalg.norm(n_all, axis=1)
        area = np.sum(area_all, axis=0)
        inds = np.zeros((n_pts - 2, 3), dtype=np.int32)
        inds[:, 1] = np.arange(n_pts - 2) + 1
        inds[:, 2] = np.arange(n_pts - 2) + 2
        c = np.sum(area_all[:,np.newaxis] * np.average(pts[inds], axis=1), axis=0)
        # for i in range(n_pts - 2):
        #     n_cur = np.cross(pts[i+1] - pts[0], pts[i+2] - pts[0]) / 2.0
        #     cur_area = np.linalg.norm(n_cur)
        #     n += n_cur
        #     area += cur_area
        #     inds = [0, i+1, i+2]
        #     c += cur_area * np.average(pts[inds], axis=0)
        c = c / area if area > 0.0 else pts[0]
        n_norm = np.linalg.norm(n)
        n = n / n_norm if n_norm > 0.0 else n
        return area, n, c

    def calc_hexahedron_volume_and_centroid(self, verts, normals, fc, areas):
        # verts: 4 pts in circular order for z_minus plain, 4 pts for z_plus in the similar order as the first 4 pts
        # normals: face normals, fc: face centers
        arithm_center = np.average(verts, axis=0)
        f_inds = np.array([[0, 2, 6, 4],
                           [1, 3, 7, 5],
                           [0, 1, 5, 4],
                           [2, 3, 7, 6],
                           [0, 1, 3, 2],
                           [4, 5, 7, 6]])
        vol = 0.0
        mc = np.zeros(3)
        for i in range(6):
            dr = arithm_center - fc[i]
            dr[np.fabs(dr / (self.eps_div + np.sqrt(areas[i]))) < 1.E-12] = 0.0
            cur_vol = np.fabs(dr.dot(normals[i])) * areas[i] / 3
            cur_mc = arithm_center / 4 + 3 / 16 * np.sum(verts[f_inds[i]], axis=0)
            mc += cur_vol * cur_mc
            vol += cur_vol
        mc = mc / vol if vol > 0.0 else cur_mc
        return vol, mc

    def calc_volumes(self):
        """
        Class method which reshapes the volumes of all the cells to a flat array (Ntot x 1)
        :return: flat volume array (Ntot x 1)
        """
        # return flat array (order='F' refers to Fortran like ordering of nodes, first index changing fastest, etc.)
        return np.reshape(self.volume, self.nodes_tot, order='F')

    def convert_to_3d_array(self, data, data_name: str):
        """
        Class method which converts the data object (scalar or vector) to a true 3D array (Nx,Ny,Nz)
        :param data: any type of data, e.g. permeability of the cells (scalar, vector, or array form)
        :param data_name: name of the data object (e.g. 'permx')
        :return data: true data 3D data array
        """
        if np.isscalar(data):
            if type(data) != int:
                data = data * np.ones(self.arr_shape, dtype=type(data))
            else:
                data = data * np.ones(self.arr_shape)
        else:
            if data.ndim == 1:
                assert data.size == self.nodes_tot, "size of %s is %s instead of %s" % (
                    data_name, data.size, self.nodes_tot)
                data = np.reshape(data, (self.nx, self.ny, self.nz),
                                  order='F')
            else:
                assert data.shape == self.arr_shape, "shape of %s is %s instead of %s" % (
                    data_name, data.shape, self.arr_shape)
        return data

    def convert_to_flat_array(self, data, data_name: str):
        """
        Class methods which converts data object from any type to true flat array of size (Ntot x 1)
        :param data: data object of any type (e.g. permeability of the cells) (scalar, vector, or 3D array form)
        :param data_name: name of the data object (e.g. 'permx')
        :return data: true flat array (Ntot x 1)
        """
        if np.isscalar(data):
            data = data * np.ones(self.nodes_tot, dtype=type(data))
        else:
            if data.ndim == 3:
                assert data.shape == self.arr_shape, "shape of %s is %s instead of %s" % (
                    data_name, data.shape, self.arr_shape)

                data = np.reshape(data, self.nodes_tot,
                                  order='F')
            elif data.ndim == 1:
                assert data.size == self.nodes_tot, "size of %s is %s instead of %s" % (
                    data_name, data.size, self.nodes_tot)
        return data

    def calc_structured_discr(self):
        """
        Class methods which performs the actual construction of the connection list
        :return cell_m: minus-side of the connection
        :return cell_p: plus-side of the connection
        :return tran: transmissibility value of connection
        :return tran_thermal: geometric coefficient of connection
        """
        print("Building connection list...")
        # Initialize zeros interface transmissibilities for each interface. Start counting from first interface in each
        # coordinate direction which is actually inside the reservoir (normally, Nx + 1 interface in the x-direction
        # if Nx is the nodes in the x-direction, but the first and the last interface are typically not neighboring any
        # other reservoir cell --> Nx - 1 "active" interfaces, i.e. connections)!
        perm_x_int = np.zeros(self.arr_shape)
        perm_y_int = np.zeros(self.arr_shape)
        perm_z_int = np.zeros(self.arr_shape)

        # Calculate interface permeability array with harmonic average of permeability in x, y and z directions:
        old_settings = np.seterr(divide='ignore',invalid='ignore')
        perm_x_int[:-1, :, :] = (self.len_cell_xdir[:-1, :, :] + self.len_cell_xdir[1:, :, :]) \
                                / (self.len_cell_xdir[:-1, :, :] / self.perm_x_cell[:-1, :, :]
                                   + self.len_cell_xdir[1:, :, :] / self.perm_x_cell[1:, :, :])

        perm_y_int[:, :-1, :] = (self.len_cell_ydir[:, :-1, :] + self.len_cell_ydir[:, 1:, :]) \
                                / (self.len_cell_ydir[:, :-1, :] / self.perm_y_cell[:, :-1, :]
                                   + self.len_cell_ydir[:, 1:, :] / self.perm_y_cell[:, 1:, :])

        perm_z_int[:, :, :-1] = (self.len_cell_zdir[:, :, :-1] + self.len_cell_zdir[:, :, 1:]) \
                                / (self.len_cell_zdir[:, :, :-1] / self.perm_z_cell[:, :, :-1]
                                   + self.len_cell_zdir[:, :, 1:] / self.perm_z_cell[:, :, 1:])

        # Calculate geometric coefficient (useful for thermal transmissibility):
        geom_coef_xdir = np.zeros(self.arr_shape)
        geom_coef_ydir = np.zeros(self.arr_shape)
        geom_coef_zdir = np.zeros(self.arr_shape)

        # Note, that for the last index for corresponding direction, geom_coef remains to be zero,
        # indicating that connection leads outside the reservoir and needs to be excluded
        geom_coef_xdir[:-1, :, :] = self.len_cell_ydir[:-1, :, :] * self.len_cell_zdir[:-1, :, :] \
                                    / (0.5 * (self.len_cell_xdir[:-1, :, :] + self.len_cell_xdir[1:, :, :]))
        geom_coef_ydir[:, :-1, :] = self.len_cell_xdir[:, :-1, :] * self.len_cell_zdir[:, :-1, :] \
                                    / (0.5 * (self.len_cell_ydir[:, :-1, :] + self.len_cell_ydir[:, 1:, :]))
        geom_coef_zdir[:, :, :-1] = self.len_cell_xdir[:, :, :-1] * self.len_cell_ydir[:, :, :-1] \
                                    / (0.5 * (self.len_cell_zdir[:, :, :-1] + self.len_cell_zdir[:, :, 1:]))

        geom_coef = np.concatenate((np.reshape(geom_coef_xdir, (self.nodes_tot), order='F'),
                                    np.reshape(geom_coef_ydir, (self.nodes_tot), order='F'),
                                    np.reshape(geom_coef_zdir, (self.nodes_tot), order='F')))

        trans_xdir = np.reshape(perm_x_int * geom_coef_xdir, (self.nodes_tot), order='F')
        trans_ydir = np.reshape(perm_y_int * geom_coef_ydir, (self.nodes_tot), order='F')
        trans_zdir = np.reshape(perm_z_int * geom_coef_zdir, (self.nodes_tot), order='F')

        # Construct connection list:
        # Store connection in x-direction:
        cell_m_x = np.array(range(trans_xdir.size), dtype=np.int32)
        cell_p_x = cell_m_x + 1

        # Store connection in y-direction:
        cell_m_y = np.array(range(trans_ydir.size), dtype=np.int32)
        cell_p_y = cell_m_y + self.nx

        # Store connection in z-direction:
        cell_m_z = np.array(range(trans_zdir.size), dtype=np.int32)
        cell_p_z = cell_m_z + self.nx * self.ny

        # glue arrays together
        cell_m = np.concatenate((cell_m_x, cell_m_y, cell_m_z))
        cell_p = np.concatenate((cell_p_x, cell_p_y, cell_p_z))
        tran = np.concatenate((trans_xdir, trans_ydir, trans_zdir))

        # mult by darcy constant
        tran *= StructDiscretizer.darcy_constant

        # Thermal transmissibility simply equals to geom_coef:
        tran_thermal = geom_coef

        # Finally extract only those entries of connection list which are inside the reservoir:
        cell_m = cell_m[geom_coef > 0]
        cell_p = cell_p[geom_coef > 0]
        tran = tran[geom_coef > 0]
        tran_thermal = tran_thermal[geom_coef > 0]

        # And apply global to local indexing (even if default)
        cell_m = self.global_to_local[cell_m]
        cell_p = self.global_to_local[cell_p]

        # Note: the filtering above could be more restrictive if 'tran > 0' were used.
        # However, thermal transmissibility is not necessarily equal to 0 everywhere, where simple transmissibility is
        # Since we decided to have the same code for thermal and non-thermal reservoirs, the relaxed filtering is
        # applied
        np.seterr(**old_settings)

        return cell_m, cell_p, tran, tran_thermal
    def calc_cpg_discr(self):
        """
        Class methods which performs the actual construction of the connection list
        :return cell_m: minus-side of the connection
        :return cell_p: plus-side of the connection
        :return tran: transmissibility value of connection
        :return tran_thermal: geometric coefficient of connection
        """
        print("Building connection list...")
        # Initialize zeros interface transmissibilities for each interface. Start counting from first interface in each
        # coordinate direction which is actually inside the reservoir (normally, Nx + 1 interface in the x-direction
        # if Nx is the nodes in the x-direction, but the first and the last interface are typically not neighboring any
        # other reservoir cell --> Nx - 1 "active" interfaces, i.e. connections)!
        perm_x_int = np.zeros(self.arr_shape)
        perm_y_int = np.zeros(self.arr_shape)
        perm_z_int = np.zeros(self.arr_shape)

        # Calculate interface permeability array with harmonic average of permeability in x, y and z directions:
        old_settings = np.seterr(divide='ignore',invalid='ignore')

        perm_x_int[:-1, :, :] = (self.len_cell_xplus[:-1, :, :] + self.len_cell_xminus[1:, :, :]) \
                                / (self.len_cell_xplus[:-1, :, :] / self.perm_xplus[:-1, :, :]
                                   + self.len_cell_xminus[1:, :, :] / self.perm_xminus[1:, :, :])
        perm_y_int[:, :-1, :] = (self.len_cell_yplus[:, :-1, :] + self.len_cell_yminus[:, 1:, :]) \
                                / (self.len_cell_yplus[:, :-1, :] / self.perm_yplus[:, :-1, :]
                                   + self.len_cell_yminus[:, 1:, :] / self.perm_yminus[:, 1:, :])
        perm_z_int[:, :, :-1] = (self.len_cell_zplus[:, :, :-1] + self.len_cell_zminus[:, :, 1:]) \
                                / (self.len_cell_zplus[:, :, :-1] / self.perm_zplus[:, :, :-1]
                                   + self.len_cell_zminus[:, :, 1:] / self.perm_zminus[:, :, 1:])

        # Calculate geometric coefficient (useful for thermal transmissibility):
        geom_coef_xdir = np.zeros(self.arr_shape)
        geom_coef_ydir = np.zeros(self.arr_shape)
        geom_coef_zdir = np.zeros(self.arr_shape)

        # Note, that for the last index for corresponding direction, geom_coef remains to be zero,
        # indicating that connection leads outside the reservoir and needs to be excluded
        geom_coef_xdir[:-1, :, :] = self.cell_data['area'][:-1, :, :, 1] / self.dist_cell_x
        geom_coef_xdir[:-1, :, :][self.dist_cell_x == 0] = 0.0
        geom_coef_ydir[:, :-1, :] = self.cell_data['area'][:, :-1, :, 3] / self.dist_cell_y
        geom_coef_ydir[:, :-1, :][self.dist_cell_y == 0] = 0.0
        geom_coef_zdir[:, :, :-1] = self.cell_data['area'][:, :, :-1, 5] / self.dist_cell_z
        geom_coef_zdir[:, :, :-1][self.dist_cell_z == 0] = 0.0

        geom_coef = np.concatenate((np.reshape(geom_coef_xdir, (self.nodes_tot), order='F'),
                                    np.reshape(geom_coef_ydir, (self.nodes_tot), order='F'),
                                    np.reshape(geom_coef_zdir, (self.nodes_tot), order='F')))

        trans_xdir = np.reshape(perm_x_int * geom_coef_xdir, (self.nodes_tot), order='F')
        trans_ydir = np.reshape(perm_y_int * geom_coef_ydir, (self.nodes_tot), order='F')
        trans_zdir = np.reshape(perm_z_int * geom_coef_zdir, (self.nodes_tot), order='F')

        # Construct connection list:
        # Store connection in x-direction:
        cell_m_x = np.array(range(trans_xdir.size), dtype=np.int32)
        cell_p_x = cell_m_x + 1

        # Store connection in y-direction:
        cell_m_y = np.array(range(trans_ydir.size), dtype=np.int32)
        cell_p_y = cell_m_y + self.nx

        # Store connection in z-direction:
        cell_m_z = np.array(range(trans_zdir.size), dtype=np.int32)
        cell_p_z = cell_m_z + self.nx * self.ny

        # glue arrays together
        cell_m = np.concatenate((cell_m_x, cell_m_y, cell_m_z))
        cell_p = np.concatenate((cell_p_x, cell_p_y, cell_p_z))
        tran = np.concatenate((trans_xdir, trans_ydir, trans_zdir))

        # mult by darcy constant
        tran *= StructDiscretizer.darcy_constant

        # Thermal transmissibility simply equals to geom_coef:
        tran_thermal = geom_coef

        # Finally extract only those entries of connection list which are inside the reservoir:
        cell_m = cell_m[geom_coef > 0]
        cell_p = cell_p[geom_coef > 0]
        tran = tran[geom_coef > 0]
        tran_thermal = tran_thermal[geom_coef > 0]

        # And apply global to local indexing (even if default)
        cell_m = self.global_to_local[cell_m]
        cell_p = self.global_to_local[cell_p]

        # Note: the filtering above could be more restrictive if 'tran > 0' were used.
        # However, thermal transmissibility is not necessarily equal to 0 everywhere, where simple transmissibility is
        # Since we decided to have the same code for thermal and non-thermal reservoirs, the relaxed filtering is
        # applied
        np.seterr(**old_settings)

        return cell_m, cell_p, tran, tran_thermal
    def calc_cell_dimensions(self, i, j, k):
        dx0 = np.zeros(2)
        dy0 = np.zeros(2)
        dz0 = np.zeros(2)
        if i != 0:
            dx0[0] = np.fabs(self.cell_data['center'][i, j, k, 0] - self.cell_data['center'][i-1, j, k, 0])
        if i != self.arr_shape[0] - 1:
            dx0[1] = np.fabs(self.cell_data['center'][i, j, k, 0] - self.cell_data['center'][i+1, j, k, 0])
        if j != 0:
            dy0[0] = np.fabs(self.cell_data['center'][i, j, k, 1] - self.cell_data['center'][i, j-1, k, 1])
        if j != self.arr_shape[1] - 1:
            dy0[1] = np.fabs(self.cell_data['center'][i, j, k, 1] - self.cell_data['center'][i, j+1, k, 1])
        if k != 0:
            dz0[0] = np.fabs(self.cell_data['center'][i, j, k, 2] - self.cell_data['center'][i, j, k-1, 2])
        if k != self.arr_shape[2] - 1:
            dz0[1] = np.fabs(self.cell_data['center'][i, j, k, 2] - self.cell_data['center'][i, j, k+1, 2])

        dx = np.sum(dx0) / np.count_nonzero(dx0) if np.count_nonzero(dx0) > 0 else 0.0
        dy = np.sum(dy0) / np.count_nonzero(dy0) if np.count_nonzero(dy0) > 0 else 0.0
        dz = np.sum(dz0) / np.count_nonzero(dz0) if np.count_nonzero(dz0) > 0 else 0.0
        return dx, dy, dz

    def apply_actnum_filter(self, actnum, cell_m, cell_p, tran, tran_thermal, arrays: list):
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        actnum = self.convert_to_flat_array(actnum, 'actnum')
        actnum[self.calc_volumes() == 0] = 0
        # Check if actnum actually has inactive cells
        n_act = actnum[actnum > 0].size
        if n_act != self.nodes_tot or \
                tran[tran < self.min_tran_tranD].size != 0 or \
                tran_thermal[tran_thermal < self.min_tran_tranD].size != 0:
            print("Applying ACTNUM...")

            print("Inactive blocks due to ACTNUM: ", actnum[actnum == 0].size)
            # Update global_to_local index
            #new_local_to_global = self.local_to_global[actnum[self.local_to_global] > 0]
            #self.global_to_local[actnum == 0] = -1
            #self.global_to_local[actnum == 1] = range(n_act)

            #compute connection list with global cell indexes
            cell_m_global = self.local_to_global[cell_m]
            cell_p_global = self.local_to_global[cell_p]

            # Create actnum filter for connections: active connections must include only active cells
            act_m = actnum[cell_m_global] != 0
            act_p = actnum[cell_p_global] != 0

            # Also filter out connections that don`t have substantial transmissibility
            act_t = tran > self.min_tran_tranD
            act_t += tran_thermal > self.min_tran_tranD
            print("Inactive connections due to transmissibility: ", act_t[act_t == False].size)
            act_conn = act_m * act_p * act_t
            print("Inactive connections total: ", act_conn[act_conn == False].size)

            # now figure which local cells (including inactive) do not participate in active connections...
            m = set(cell_m[act_conn])
            p = set(cell_p[act_conn])
            all = set(range(self.nodes_tot))
            not_connected = all.difference(m)
            not_connected = list(not_connected.difference(p))

            print("Inactive blocks due to inactive connections: ", len(not_connected) - actnum[actnum == 0].size)

            # and make corresponding global cells inactive too
            actnum[self.local_to_global[not_connected]] = 0
            n_act = actnum[actnum > 0].size

            # Now filter local cells, removing those which correspond to inactive global cells
            # This way self.local_to_global array will be squeezed to contain only active local cells,
            # preserving correct global indexes for them
            new_local_to_global = self.local_to_global[actnum[self.local_to_global] != 0]
            self.local_to_global = new_local_to_global
            # update global_to_local index for inactive cells
            self.global_to_local[actnum == 0] = -1
            # update global_to_local index for active cells
            self.global_to_local[new_local_to_global] = np.arange(n_act, dtype=np.int32)


            # re-index active connections using updated local indexes
            cell_m_local = self.global_to_local[cell_m_global[act_conn]]
            cell_p_local = self.global_to_local[cell_p_global[act_conn]]
            tran_local = tran[act_conn]
            tran_thermal_local = tran_thermal[act_conn]

        else:
            cell_m_local = cell_m
            cell_p_local = cell_p
            tran_local = tran
            tran_thermal_local = tran_thermal

        # Apply actnum filter, if any, and global_to_local indexing to arrays
        arrays_local = []
        for i, a in enumerate(arrays):
            a = self.convert_to_flat_array(a, 'Unknown')
            arrays_local.append(a[self.local_to_global])
        np.seterr(**old_settings)
        return cell_m_local, cell_p_local, tran_local, tran_thermal_local, arrays_local

    def calc_well_index(self, i, j, k, well_radius=0.1524, segment_direction='z_axis', skin=0):
        """
        Class method which construct the well index for each well segment/perforation
        :param i: "human" counting of x-location coordinate of perforation
        :param j: "human" counting of y-location coordinate of perforation
        :param k: "human" counting of z-location coordinate of perforation
        :param well_radius: radius of the well-bore
        :param segment_direction: direction in which the segment perforates the reservoir block
        :param skin: skin factor for pressure loss around well-bore due to formation damage
        :return well_index: well-index of particular perforation
        """
        assert (i > 0), "Perforation block coordinate should be positive"
        assert (j > 0), "Perforation block coordinate should be positive"
        assert (k > 0), "Perforation block coordinate should be positive"
        assert (i <= self.nx), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (j <= self.ny), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (k <= self.nz), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        i -= 1
        j -= 1
        k -= 1

        # compute reservoir block index
        res_block = k * self.nx * self.ny + j * self.nx + i

        # check if target grid block is active
        if self.global_to_local[res_block] > -1:

            # Store grid-dimensions of segmet and permeability:
            if self.is_cpg:
                dx, dy, dz = self.calc_cell_dimensions(i, j, k)
            else:
                dx = self.len_cell_xdir[i, j, k]
                dy = self.len_cell_ydir[i, j, k]
                dz = self.len_cell_zdir[i, j, k]
            kx = self.perm_x_cell[i, j, k]
            ky = self.perm_y_cell[i, j, k]
            kz = self.perm_z_cell[i, j, k]

            well_index = 0
            if segment_direction == 'z_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kx) * dx ** 2 + np.sqrt(kx / ky) * dy ** 2) / \
                               ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * ky) / (np.log(peaceman_rad / well_radius) + skin)
                if kx == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'x_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kz) * dz ** 2 + np.sqrt(kz / ky) * dy ** 2) / \
                               ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kz * ky) / (np.log(peaceman_rad / well_radius) + skin)
                if kz == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'y_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(kz / kx) * dx ** 2 + np.sqrt(kx / kz) * dz ** 2) / \
                               ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * kz) / (np.log(peaceman_rad / well_radius) + skin)
                if kx == 0 or kz == 0: well_index = 0.0

            well_index = well_index * StructDiscretizer.darcy_constant
        else:
            well_index = 0

        return self.global_to_local[res_block], well_index

    @staticmethod
    def dump_connection_list(cell_m, cell_p, conn, filename):
        """
        Static method which dumps the connection list to a file
        :param cell_m: minus-side of the connection
        :param cell_p: plus-side of the connection
        :param conn: transmisibility value of connection
        :param filename: name of the desired file
        :return:
        """
        with open(filename, 'w') as f:
            f.write('TPFACONNS\n')
            f.write('%d\n' % cell_m.size)
            for i, m in enumerate(cell_m):
                f.write('%d\t%d\t%.15f\n' % (m, cell_p[i], conn[i]))
            f.write('/' % cell_m.size)
        return 0
