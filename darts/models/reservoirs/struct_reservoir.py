from math import pi

import numpy as np
from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node, value_vector, index_vector
from darts.mesh.struct_discretizer import StructDiscretizer
from darts.tools.pyevtk import hl
from scipy.interpolate import griddata
import os


class StructReservoir:
    def __init__(self, timer, nx: int, ny: int, nz: int,
                 dx, dy, dz,
                 permx, permy, permz,
                 poro, depth, actnum=1, global_to_local = 0, op_num=0, coord=0, zcorn=0):

        """
        Class constructor method
        :param timer: timer object to measure discretization time
        :param nx: number of reservoir blocks in the x-direction
        :param ny: number of reservoir blocks in the y-direction
        :param nz: number of reservoir blocks in the z-direction
        :param dx: size of the reservoir blocks in the x-direction (scalar or vector form) [m]
        :param dy: size of the reservoir blocks in the y-direction (scalar or vector form) [m]
        :param dz: size of the reservoir blocks in the z-direction (scalar or vector form) [m]
        :param permx: permeability of the reservoir blocks in the x-direction (scalar or vector form) [mD]
        :param permy: permeability of the reservoir blocks in the y-direction (scalar or vector form) [mD]
        :param permz: permeability of the reservoir blocks in the z-direction (scalar or vector form) [mD]
        :param poro: porosity of the reservoir blocks
        :param actnum: attribute of activity of the reservoir blocks (all are active by default)
        :param global_to_local: one can define arbitrary indexing (mapping from global to local) for local
                                arrays. Default indexing is by X (fastest),then Y, and finally Z (slowest)
        :param op_num: index of required operator set of the reservoir blocks (the first by default).
                       Use to introduce PVTNUM, SCALNUM, etc.
        :param coord: COORD keyword values for more accurate geometry during VTK export (no values by default)
        :param zcron: ZCORN keyword values for more accurate geometry during VTK export (no values by default)

        """

        self.timer = timer
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.coord = coord
        self.zcorn = zcorn
        self.permx = permx
        self.permy = permy
        self.permz = permz
        self.n = nx * ny * nz
        self.global_data = {'dx': dx,
                            'dy': dy,
                            'dz': dz,
                            'permx': permx,
                            'permy': permy,
                            'permz': permz,
                            'poro': poro,
                            'depth': depth,
                            'actnum': actnum,
                            'op_num': op_num,
                            }
        self.discretizer = StructDiscretizer(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, permx=permx, permy=permy,
                                             permz=permz, global_to_local = global_to_local)

        self.timer.node['initialization'].node['connection list generation'] = timer_node()
        self.timer.node['initialization'].node['connection list generation'].start()
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_structured_discr()
        self.timer.node['initialization'].node['connection list generation'].stop()

        volume = self.discretizer.calc_volumes()
        self.global_data['volume'] = volume

        # apply actnum filter if needed - all arrays providing a value for a single grid block should be passed
        arrs = [poro, depth, volume, op_num]
        cell_m, cell_p, tran, tran_thermal, arrs_local = self.discretizer.apply_actnum_filter(actnum, cell_m,
                                                                                              cell_p, tran,
                                                                                              tran_thermal, arrs)
        poro, depth, volume, op_num = arrs_local
        self.global_data['global_to_local'] = self.discretizer.global_to_local
        # create mesh object
        self.mesh = conn_mesh()

        # Initialize mesh using built connection list
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran),
                       value_vector(tran_thermal))

        # taking into account actnum
        self.nb = volume.size

        # Create numpy arrays wrapped around mesh data (no copying)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.op_num = np.array(self.mesh.op_num, copy=False)
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.rcond = np.array(self.mesh.rock_cond, copy=False)

        self.poro[:] = poro
        self.depth[:] = depth
        self.volume[:] = volume
        self.op_num[:] = op_num

        self.wells = []

        self.vtk_z = 0
        self.vtk_y = 0
        self.vtk_x = 0
        self.vtk_filenames_and_times = {}
        self.vtkobj = 0

        if np.isscalar(self.coord):
            # Usual structured grid generated from DX, DY, DZ, DEPTH
            self.vtk_grid_type = 0
        else:
            # CPG grid from COORD ZCORN
            self.vtk_grid_type = 1


    def set_boundary_volume(self, xy_minus=-1, xy_plus=-1, yz_minus=-1, yz_plus=-1, xz_minus=-1, xz_plus=-1):
        # get 3d shape
        volume = self.discretizer.volume

        # apply changes
        if xy_minus > -1:
            volume[:, :, 0] = xy_minus
        if xy_plus > -1:
            volume[:, :, -1] = xy_plus
        if yz_minus > -1:
            volume[0, :, :] = yz_minus
        if yz_plus > -1:
            volume[-1, :, :] = yz_plus
        if xz_minus > -1:
            volume[:, 0, :] = xz_minus
        if xz_plus > -1:
            volume[:, -1, :] = xz_plus
        # reshape to 1d
        volume = np.reshape(volume, self.discretizer.nodes_tot, order='F')
        # apply actnum and assign to mesh.volume
        self.volume[:] = volume[self.discretizer.local_to_global]

    def add_well(self, name, wellbore_diameter=0.15):
        well = ms_well()
        well.name = name
        # so far put only area here,
        # to be multiplied by segment length later

        well.segment_volume = pi * wellbore_diameter ** 2 / 4

        # also to be filled up when the first perforation is made
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_depth_increment = 0
        self.wells.append(well)
        return well

    def add_perforation(self, well, i, j, k, well_radius=0.1524, well_index=-1, segment_direction='z_axis', skin=0,
                        multi_segment=True,
                        verbose=False):
        # calculate well index and get local index of reservoir block
        res_block_local, wi = self.discretizer.calc_well_index(i=i, j=j, k=k, well_radius=well_radius,
                                                               segment_direction=segment_direction,
                                                               skin=skin)

        if well_index == -1:
            well_index = wi

        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        # add completion only if target block is active
        if res_block_local > -1:
            if len(well.perforations) == 0:
                well.well_head_depth = self.depth[res_block_local]
                well.well_body_depth = well.well_head_depth
                well.segment_depth_increment = self.discretizer.len_cell_zdir[i - 1, j - 1, k - 1]
                well.segment_volume *= well.segment_depth_increment
            well.perforations = well.perforations + [(well_block, res_block_local, well_index)]
            if verbose:
                print('Added perforation for well %s to block %d [%d, %d, %d] with WI=%f' % (
                    well.name, res_block_local, i, j, k, well_index))
        else:
            if verbose:
                print('Neglected perforation for well %s to block [%d, %d, %d] (inactive block)' % (well.name, i, j, k))

    def init_wells(self):
        self.mesh.add_wells(ms_well_vector(self.wells))
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()

    def export_vtk(self, file_name, t, local_cell_data, global_cell_data, export_constant_data=True):

        nb = self.discretizer.nodes_tot
        cell_data = global_cell_data.copy()

        # only for the first export call
        if len(self.vtk_filenames_and_times) == 0:
            if self.vtk_grid_type == 0:
                if (self.n == self.nx) or (self.n == self.ny) or (self.n == self.nz):
                    self.generate_vtk_grid(compute_depth_by_dz_sum=False) # Add this (if condition) for special 1D vtk export
                else:
                    self.generate_vtk_grid()
            else:
                self.generate_cpg_vtk_grid()
            self.vtk_path = './vtk_data/'
            if len(self.vtk_filenames_and_times) == 0:
                os.makedirs(self.vtk_path, exist_ok=True)

            if export_constant_data:
                mesh_geom_dtype = np.float32
                for key, data in self.global_data.items():
                    if np.isscalar(data):
                        if type(data) == int:
                            data = data * np.ones(nb, dtype=int)
                        else:
                            data = data * np.ones(nb, dtype=mesh_geom_dtype)
                    cell_data[key] = data

        vtk_file_name = self.vtk_path + file_name + '_ts%d' % len(self.vtk_filenames_and_times)

        for key, value in local_cell_data.items():
            global_array = np.ones(nb, dtype=value.dtype) * np.nan
            global_array[self.discretizer.local_to_global] = value
            cell_data[key] = global_array

        if self.vtk_grid_type == 0:
            vtk_file_name = hl.gridToVTK(vtk_file_name, self.vtk_x, self.vtk_y, self.vtk_z, cellData=cell_data)
        else:
            for key, value in cell_data.items():
                self.vtkobj.AppendScalarData(key, cell_data[key])

            vtk_file_name = self.vtkobj.Write2VTU(vtk_file_name)
            if len(self.vtk_filenames_and_times) == 0:
                for key, data in self.global_data.items():
                    self.vtkobj.VTK_Grids.GetCellData().RemoveArray(key)
                self.vtkobj.VTK_Grids.GetCellData().RemoveArray('cellNormals')

        # in order to have correct timesteps in Paraview, write down group file
        # since the library in use (pyevtk) requires the group file to call .save() method in the end,
        # and does not support reading, track all written files and times and re-write the complete
        # group file every time

        self.vtk_filenames_and_times[vtk_file_name] = t

        self.group = hl.VtkGroup(file_name)
        for fname, t in self.vtk_filenames_and_times.items():
            self.group.addFile(fname, t)
        self.group.save()


    def generate_vtk_grid(self, strict_vertical_layers=True, compute_depth_by_dz_sum=True):
        # interpolate 2d array using grid (xx, yy) and specified method
        def interpolate_slice(xx, yy, array, method):
            array = np.ma.masked_invalid(array)
            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]
            array = griddata((x1, y1), newarr.ravel(),
                             (xx, yy),
                             method=method)
            return array

        def interpolate_zeroes_2d(array):
            array[array == 0] = np.nan
            x = np.arange(0, array.shape[1])
            y = np.arange(0, array.shape[0])
            xx, yy = np.meshgrid(x, y)

            # stage 1 - fill in interior data using cubic interpolation
            array = interpolate_slice(xx, yy, array, 'cubic')
            # stage 2 - fill exterior data using nearest
            array = interpolate_slice(xx, yy, array, 'nearest')
            return array

        def interpolate_zeroes_3d(array_3d):
            if array_3d[array_3d == 0].size > 0:
                array_3d[array_3d == 0] = np.nan
                x = np.arange(0, array_3d.shape[1])
                y = np.arange(0, array_3d.shape[0])
                xx, yy = np.meshgrid(x, y)
                # slice array over third dimension
                for k in range(array_3d.shape[2]):
                    array = array_3d[:, :, k]
                    if array[np.isnan(array) == False].size > 3:
                        # stage 1 - fill in interior data using cubic interpolation
                        array = interpolate_slice(xx, yy, array, 'cubic')

                    if array[np.isnan(array) == False].size > 0:
                        # stage 2 - fill exterior data using nearest
                        array_3d[:, :, k] = interpolate_slice(xx, yy, array, 'nearest')
                    else:
                        if k > 0:
                            array_3d[:, :, k] = np.mean(array_3d[:, :, k - 1])
                        else:
                            array_3d[:, :, k] = np.mean(array_3d)

            return array_3d

        nx = self.discretizer.nx
        ny = self.discretizer.ny
        nz = self.discretizer.nz

        # consider 16-bit float is enough for mesh geometry
        mesh_geom_dtype = np.float32

        # get tops from depths
        if np.isscalar(self.global_data['depth']):
            tops = self.global_data['depth'] * np.ones((nx, ny))
            compute_depth_by_dz_sum = True
        elif compute_depth_by_dz_sum:
            tops = self.global_data['depth'][:nx * ny]
            tops = np.reshape(tops, (nx, ny), order='F').astype(mesh_geom_dtype)
        else:
            depths = np.reshape(self.global_data['depth'], (nx, ny, nz), order='F').astype(mesh_geom_dtype)

        # tops_avg = np.mean(tops[tops > 0])
        # tops[tops <= 0] = 2000

        # average x-s of the left planes for the left cross-section (i=1)
        lefts = 0 * np.ones((ny, nz))
        # average y-s of the front planes for the front cross_section (j=1)
        fronts = 0 * np.ones((nx, nz))

        self.vtk_x = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)
        self.vtk_y = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)
        self.vtk_z = np.zeros((nx + 1, ny + 1, nz + 1), dtype=mesh_geom_dtype)

        if compute_depth_by_dz_sum:
            tops = interpolate_zeroes_2d(tops)
            tops_padded = np.pad(tops, 1, 'edge')
        else:
            depths_padded = np.pad(depths, 1, 'edge').astype(mesh_geom_dtype)
        lefts_padded = np.pad(lefts, 1, 'edge')
        fronts_padded = np.pad(fronts, 1, 'edge')

        dx_padded = np.pad(self.discretizer.len_cell_xdir, 1, 'edge').astype(mesh_geom_dtype)
        dy_padded = np.pad(self.discretizer.len_cell_ydir, 1, 'edge').astype(mesh_geom_dtype)
        dz_padded = np.pad(self.discretizer.len_cell_zdir, 1, 'edge').astype(mesh_geom_dtype)

        if strict_vertical_layers:
            print("Interpolating missing data in DX...")
            dx_padded_top = interpolate_zeroes_2d(dx_padded[:, :, 0])
            dx_padded = np.repeat(dx_padded_top[:, :, np.newaxis], dx_padded.shape[2], axis=2)

            print("Interpolating missing data in DY...")
            dy_padded_top = interpolate_zeroes_2d(dy_padded[:, :, 0])
            dy_padded = np.repeat(dy_padded_top[:, :, np.newaxis], dy_padded.shape[2], axis=2)
        else:
            print("Interpolating missing data in DX...")
            interpolate_zeroes_3d(dx_padded)
            print("Interpolating missing data in DY...")
            interpolate_zeroes_3d(dy_padded)

        # DZ=0 can actually be correct values in case of zero-thickness inactive blocks
        # So we don`t need to interpolate them

        #print("Interpolating missing data in DZ...")
        #interpolate_zeroes_3d(dz_padded)

        if not compute_depth_by_dz_sum:
            print("Interpolating missing data in DEPTH...")
            interpolate_zeroes_3d(depths_padded)

        # initialize k=0 as sum of 4 neighbours
        if compute_depth_by_dz_sum:
            self.vtk_z[:, :, 0] = (tops_padded[:-1, :-1] +
                                   tops_padded[:-1, 1:] +
                                   tops_padded[1:, :-1] +
                                   tops_padded[1:, 1:]) / 4
        else:
            self.vtk_z[:, :, 0] = (depths_padded[:-1, :-1, 0] - dz_padded[:-1, :-1, 0] / 2 +
                                   depths_padded[:-1, 1:, 0] - dz_padded[:-1, 1:, 0] / 2 +
                                   depths_padded[1:, :-1, 0] - dz_padded[1:, :-1, 0] / 2 +
                                   depths_padded[1:, 1:, 0] - dz_padded[1:, 1:, 0] / 2) / 4
        # initialize i=0
        self.vtk_x[0, :, :] = (lefts_padded[:-1, :-1] +
                               lefts_padded[:-1, 1:] +
                               lefts_padded[1:, :-1] +
                               lefts_padded[1:, 1:]) / 4
        # initialize j=0
        self.vtk_y[:, 0, :] = (fronts_padded[:-1, :-1] +
                               fronts_padded[:-1, 1:] +
                               fronts_padded[1:, :-1] +
                               fronts_padded[1:, 1:]) / 4

        # assign the rest coordinates by averaged size of neigbouring cells
        if compute_depth_by_dz_sum:
            self.vtk_z[:, :, 1:] = (dz_padded[:-1, :-1, 1:-1] +
                                    dz_padded[:-1, 1:, 1:-1] +
                                    dz_padded[1:, :-1, 1:-1] +
                                    dz_padded[1:, 1:, 1:-1]) / 4
        else:
            self.vtk_z[:, :, 1:] = (depths_padded[:-1, :-1, 1:-1] + dz_padded[:-1, :-1, 1:-1] / 2 +
                                    depths_padded[:-1, 1:, 1:-1] + dz_padded[:-1, 1:, 1:-1] / 2 +
                                    depths_padded[1:, :-1, 1:-1] + dz_padded[1:, :-1, 1:-1] / 2 +
                                    depths_padded[1:, 1:, 1:-1] + dz_padded[1:, 1:, 1:-1] / 2) / 4

        self.vtk_x[1:, :, :] = (dx_padded[1:-1, :-1, :-1] +
                                dx_padded[1:-1, :-1, 1:] +
                                dx_padded[1:-1, 1:, :-1] +
                                dx_padded[1:-1, 1:, 1:]) / 4

        self.vtk_y[:, 1:, :] = (dy_padded[:-1, 1:-1, :-1] +
                                dy_padded[:-1, 1:-1, 1:] +
                                dy_padded[1:, 1:-1, :-1] +
                                dy_padded[1:, 1:-1, 1:]) / 4

        self.vtk_x = np.cumsum(self.vtk_x, axis=0)
        self.vtk_y = np.cumsum(self.vtk_y, axis=1)
        if compute_depth_by_dz_sum:
            self.vtk_z = np.cumsum(self.vtk_z, axis=2)

        # convert to negative coordinate
        z_scale = -1
        self.vtk_z *= z_scale

    def generate_cpg_vtk_grid(self):
        from darts.tools import GRDECL2VTK

        self.vtkobj = GRDECL2VTK.GeologyModel()
        self.vtkobj.GRDECL_Data.COORD = self.coord
        self.vtkobj.GRDECL_Data.ZCORN = self.zcorn
        self.vtkobj.GRDECL_Data.NX = self.nx
        self.vtkobj.GRDECL_Data.NY = self.ny
        self.vtkobj.GRDECL_Data.NZ = self.nz
        self.vtkobj.GRDECL_Data.N = self.n
        self.vtkobj.GRDECL_Data.GRID_type = 'CornerPoint'
        self.vtkobj.GRDECL2VTK()
        #self.vtkobj.decomposeModel()
