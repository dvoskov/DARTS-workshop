# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
import time
from .transcalc import *


# Class definition of the StructDiscretizer, which discretizes a structured reservoir
class StructDiscretizer:
    # Define Darcy constant for changes to correct units:
    darcy_constant = 0.0085267146719160104986876640419948

    def __init__(self, nx, ny, nz, dx, dy, dz, permx, permy, permz, global_to_local = 0):
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

        # If scalar dx, dy, and dz are specified: Store constant control volume dimensions
        self.len_cell_xdir = self.convert_to_3d_array(dx, 'dx')
        self.len_cell_ydir = self.convert_to_3d_array(dy, 'dy')
        self.len_cell_zdir = self.convert_to_3d_array(dz, 'dz')
        self.perm_x_cell = self.convert_to_3d_array(permx, 'permx')
        self.perm_y_cell = self.convert_to_3d_array(permy, 'permy')
        self.perm_z_cell = self.convert_to_3d_array(permz, 'permz')
        self.volume = self.len_cell_xdir * self.len_cell_ydir * self.len_cell_zdir

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
            data = data * np.ones(self.arr_shape, dtype=type(data))
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
        i -= 1
        j -= 1
        k -= 1

        # compute reservoir block index
        res_block = k * self.nx * self.ny + j * self.nx + i

        # check if target grid block is active
        if self.global_to_local[res_block] > -1:

            # Store grid-dimensions of segmet and permeability:
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

            elif segment_direction == 'x_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kz) * dz ** 2 + np.sqrt(kz / ky) * dy ** 2) / \
                               ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kz * ky) / (np.log(peaceman_rad / well_radius) + skin)

            elif segment_direction == 'y_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(kz / kx) * dx ** 2 + np.sqrt(kx / kz) * dz ** 2) / \
                               ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * kz) / (np.log(peaceman_rad / well_radius) + skin)

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
