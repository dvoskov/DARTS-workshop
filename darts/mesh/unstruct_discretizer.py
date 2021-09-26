# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
import subprocess
import time
from itertools import combinations, compress
import meshio
import numpy as np
from scipy.linalg import null_space
from .geometrymodule import Hexahedron, Wedge, Tetrahedron, Pyramid, Quadrangle, Triangle, Face, FType
from .transcalc import TransCalculations
import copy
import os
import pickle


""""
    Unstructured discretization class.
    Some definitions:
        - Nodes:    Vertices, points
        - Cells:    Control volumes
        - Face:     Sides of the control volume
        
    Finding intersections in this unstructured discretization module are down in the following way (2D example):
         "Grid example"
          0----1-----2-3
          | a / \ c /  |
          |  / b \ / d |
          4-5-----6----7 
    Example: we are investigating interface from node 1 to node 6 (with neighboring cells b and c):
        Have two sets: 
            - cells belonging to node 1: {a, b, c}
            - cells belonging to node 5: {b, c, d}
        Find intersection of two sets, which is also a set: {b,c}
        When length of the set is exactly 2 --> interface has two neighboring cells and therefore a connection has to 
        be added to the connection list of the reservoir (when having in-active cells, need to add one additional check)
    This example easily generalizes to 3D and any interface geometry (if cells are conformal)!
    Also, fractures work in an analogues way!
"""


# Definitions for the unstructured discretization class:
class UnstructDiscretizer:
    def __init__(self, permx, permy, permz, frac_aper, mesh_file: str, num_matrix_cells=0, num_fracture_cells=0):
        """
        Class constructor method
        :param permx: permeability data object (either in scalar or vector form) in x-direction
        :param permy: permeability data object (either in scalar or vector form) in y-direction
        :param permz: permeability data object (either in scalar or vector form) in z-direction
        :param frac_aper: fracture aperture data object (either in scalar or vector form)
        :param mesh_file: name of the mesh file (in string form)
        :param num_matrix_cells: number of matrix cells, if known before hand! (in scalar form)
        :param num_fracture_cells: number of fracture cells, if known before hand! (in scalar form)
        """
        self.mesh_file = mesh_file  # Name of the input meshfile
        self.mesh_data = []  # Initialize empty mesh data list
        self.mat_cell_info_dict = {}  # Dictionary containing information of all the matrix cells
        self.frac_cell_info_dict = {}  # Dictionary containing information of all the fracture cells
        self.bound_cell_info_dict = {}  # Dictionary containing information of all the boundary cells
        self.output_face_info_dict = {} # Dictionary containing information of all the faces for output
        self.mat_cells_to_node = {}  # Dictionary containing all the cells belonging to each matrix node
        self.frac_cells_to_node = {}  # Dictionary containing all the cells belonging to each fracture node
        self.bound_cells_to_node = {}  # Dictionary containing all the cells belonging to each fracture node
        self.geometries_in_mesh_file = []  # List with geometries found in mesh file
        self.frac_geometries_in_file = []  # List with fracture geometries found in mesh file
        self.matrix_cell_count = 0  # Number of matrix cells found when calc. matrix cell information
        self.fracture_cell_count = 0  # Number of fracture cells found when calc. fracture cell information
        self.bound_cell_count = 0  # Number of boundary cells
        self.output_face_count = 0 # Number of faces for output
        self.mat_cells_tot = num_matrix_cells  # Total number of matrix cells
        self.frac_cells_tot = num_fracture_cells  # Total number of fracture cells
        self.output_face_tot = 0  # Number of output faces
        self.volume_all_cells = {}  # Volume of matrix and fracture cells (later as numpy.array)
        self.depth_all_cells = {}  # Depth of matrix and fracture cells (later as numpy.array)
        self.centroid_all_cells = {}  # Centroid of matrix and fracture cells (later as numpy.array)

        self.bound_cells_to_node = {}
        self.boundary_connections = {}
        self.tol = 1.E-10 # Tolerance in MPxA
        self.mpfa_connections = {}  # Pairs of cells which has a common sub-interface per node (int. region)
        self.mpsa_connections = {} # Stress flux connections
        self.n_dim = 0 # Mesh dimension (2 for 2D, 3 for 3D)
        self.mpfa_connections_num = 0 # Number of MPFA connections
        self.mpsa_connections_num = 0 # Number of MPSA connections
        self.boundary_conditions = {}
        self.physical_tags = {}
        self.disp_gradients = {}    # Displacement gradient & corresponding stencil for every matrix cell
        self.ith_iter = 0
        self.Ft_prev = {}
        self.Ft_iter = {}
        self.Fharm = {}
        self.Ft1 = {}
        self.Ft2 = {}
        self.Fcont1 = {}
        self.Fcont2 = {}

        # Store the currently available matrix and fracture geometries supported by this unstructured discretizer:
        self.available_matrix_geometries = {'hexahedron', 'wedge', 'tetra', 'pyramid'}
        self.available_fracture_geometries = {'quad', 'triangle'}

        # Permeability data (perm{x,y,z}) can come in two forms:
        #   scalar:                 kx,y,z=cte
        #   vector (N_mat x 1):     kx,y,z are variable on domain (for matrix cells)
        self.perm_x_cell = self.check_matrix_data_input(permx, 'permx')
        self.perm_y_cell = self.check_matrix_data_input(permy, 'permx')
        self.perm_z_cell = self.check_matrix_data_input(permz, 'permx')

        # Convery fracture aperture to permeability by two types:
        #   scalar:                 constant aperature for all fractures
        #   vector (N_frac x 1):    variable aperture for each fracture segment
        self.fracture_aperture = self.check_fracture_data_input(frac_aper, 'frac_aper')
        self.perm_frac_cell = (self.fracture_aperture ** 2) / 12 * 1E15

    def init_matrix_stiffness(self, props):
        self.stiffness = {}
        self.stf = {}
        self.E = {}
        self.nu = {}
        for id, prop in props.items():
            self.E[id] = prop['E']
            self.nu[id] = prop['nu']
            self.stiffness[id] = self.get_isotropic_stiffness(prop['E'], prop['nu'])
            self.stf[id] = self.get_stiffness_submatrices(self.stiffness[id])
    def init_matrix_stiffness_by_value(self, props):
        self.stiffness = {}
        self.stf = {}
        for id, prop in props.items():
            self.stiffness[id] = prop
            self.stf[id] = self.get_stiffness_submatrices(prop)

    def check_matrix_data_input(self, data, data_name: str):
        """
        Class method which checks the input data for matrix cells
        :param data: scalar or vector with data
        :param data_name: string which represents the data
        :return: correct data object size
        """
        if self.mat_cells_tot > 0:
            if np.isscalar(data):
                # Input data object is scalar value
                data = data * np.ones((self.mat_cells_tot,), dtype=type(data))
                # ONLY UNCOMMENT THIS LINE BELOW IF YOU WANT HETEROGENEITY BY RANDOM INPUT:
                # data = data * np.random.uniform(0.95, 1.05, (self.mat_cells_tot,))
        return data

    def check_fracture_data_input(self, data, data_name: str):
        """
        Class method which checks the input data for fracture cells
        :param data: scalar or vector with data
        :param data_name: string which represents the data
        :return: correct data object size
        """
        if self.frac_cells_tot > 0:
            if np.isscalar(data):
                # Input data object is scalar value
                data = data * np.ones((self.frac_cells_tot,), dtype=type(data))
        return data

    def load_mesh(self, cache=0):
        """"
        Class method which loads the mesh data of a specified file, using the module meshio module (third party).
        """
        start_time_module = time.time()
        read_from_cache = False

        if cache:
            if os.path.isfile(self.mesh_file + '.meshObject.cache'):
                print('Start loading mesh from cache file (pickle)...')
                with open(self.mesh_file + '.meshObject.cache', 'rb') as handle:
                    self.meshObject = pickle.load(handle)

                self.mesh_data = self.meshObject['mesh_data']
                self.mat_cells_tot = self.meshObject['mat_cells_tot']
                self.frac_cells_tot = self.meshObject['frac_cells_tot']
                read_from_cache = True
                print('Time to load Mesh: {:f} [sec]'.format((time.time() - start_time_module)))
                print(self.mesh_data)

        if not read_from_cache:
            print('Start loading mesh from mesh file...')
            self.mesh_data = meshio.read(self.mesh_file)

            # Store all available geometries of the objects found by meshio in a list:
            for ith_geometry in self.mesh_data.cells_dict:
                self.geometries_in_mesh_file.append(ith_geometry)

                if ith_geometry in self.available_fracture_geometries:
                    self.frac_geometries_in_file.append(ith_geometry)

            # Create default dictionary entry:
            for geometry in self.available_matrix_geometries:
                self.mesh_data.cells_dict.setdefault(geometry, np.array([]))

            for geometry in self.available_fracture_geometries:
                self.mesh_data.cells_dict.setdefault(geometry, np.array([]))

            print('Time to load Mesh: {:f} [sec]'.format((time.time() - start_time_module)))
            print(self.mesh_data)

            # If matrix cells have not yet been specified (prior to loading mesh), then count them here:
            count_mat_cells = False
            count_frac_cells = False
            if self.mat_cells_tot == 0:
                count_mat_cells = True
            if self.frac_cells_tot == 0:
                count_frac_cells = True

            for ith_geometry in self.mesh_data.cells_dict:
                # Find and store number of cells:
                if ith_geometry in self.available_fracture_geometries:
                    # Geometry indicates (supported) fracture type geometry (2D element):
                    if count_frac_cells:
                        self.frac_cells_tot += self.mesh_data.cells_dict[ith_geometry].shape[0]
                elif ith_geometry in self.available_matrix_geometries:
                    # Geometry indicates (supported) matrix type geometry (3D element):
                    if count_mat_cells:
                        self.mat_cells_tot += self.mesh_data.cells_dict[ith_geometry].shape[0]
                else:
                    # Found geometry which is not supported by discretizer!
                    print('!!!!!!!!!!!!UNSUPORTED GEOMETRY FOUND!!!!!!!!!!!!')
            print('Total number of matrix cells found: {:d}'.format(self.mat_cells_tot))
            print('Total number of fracture cells found: {:d}'.format(self.frac_cells_tot))
            print('------------------------------------------------\n')

        # Interpret input perm_data:
        self.perm_x_cell = self.check_matrix_data_input(self.perm_x_cell, 'permx')
        self.perm_y_cell = self.check_matrix_data_input(self.perm_y_cell, 'permy')
        self.perm_z_cell = self.check_matrix_data_input(self.perm_z_cell, 'permz')
        self.fracture_aperture = self.check_fracture_data_input(self.fracture_aperture, 'frac_aper')
        self.perm_frac_cell = self.check_fracture_data_input(self.perm_frac_cell, 'perm_frac')

        if cache and not read_from_cache:
            self.meshObject = {}
            self.meshObject['mesh_data'] = self.mesh_data
            self.meshObject['mat_cells_tot'] = self.mat_cells_tot
            self.meshObject['frac_cells_tot'] = self.frac_cells_tot

            with open(self.mesh_file + '.meshObject.cache', 'wb') as handle:
                pickle.dump(self.meshObject, handle, protocol=4)
            print("Files have been read and cached.")
        return 0

    def load_mesh_with_bounds(self):
        start_time_module = time.time()
        print('Start loading mesh...')
        self.mesh_data = meshio.read(self.mesh_file)

        # Count all the cells, boundary cells and fractures by their types
        self.mat_cells_tot = self.bound_cells_tot = self.frac_cells_tot = 0
        for geometry, all_types in self.mesh_data.cell_data.items():
            types = all_types['gmsh:physical']
            unique, counts = np.unique(types, return_counts=True)
            for type, count in zip(unique, counts):
                if type in self.physical_tags['boundary']:
                    self.bound_cells_tot += count
                elif type in self.physical_tags['matrix']:
                    self.mat_cells_tot += count
                elif type in self.physical_tags['fracture']:
                    self.frac_cells_tot += count
                elif type in self.physical_tags['output']:
                    self.output_face_tot += count
                else:
                    # Found geometry which is not supported by discretizer!
                    print('!!!!!!!!!!!!UNSUPORTED GEOMETRY FOUND!!!!!!!!!!!!')

        self.perm_x_cell = self.check_matrix_data_input(self.perm_x_cell, 'permx')
        self.perm_y_cell = self.check_matrix_data_input(self.perm_y_cell, 'permy')
        self.perm_z_cell = self.check_matrix_data_input(self.perm_z_cell, 'permz')

        # Find cells belonging to particular node:
        cell_count = 0
        mat_count = 0
        face_count = 0
        bound_count = 0
        global_count = 0
        output_count = 0

        for geometry, types in sorted(list(self.mesh_data.cell_data.items()), key=lambda x: -x[1]['gmsh:physical'][0]):
            tags = types['gmsh:physical']
            # Main loop over different existing geometries
            for ith_cell, nodes_to_cell in enumerate(self.mesh_data.cells[geometry]):
                global_count += 1

                # Calculate general information for cell, based on geometry, nodes belong to cell and their coordinates:
                # matrix cells
                if tags[ith_cell] in self.physical_tags['matrix']:
                    for key in nodes_to_cell:
                        self.mat_cells_to_node.setdefault(key, [])
                        self.mat_cells_to_node[key].append(cell_count)

                    if geometry == 'hexahedron':
                        self.mat_cell_info_dict[cell_count] = \
                            Hexahedron(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                       np.array([self.perm_x_cell[mat_count],
                                                 self.perm_y_cell[mat_count],
                                                 self.perm_z_cell[mat_count]]),
                                       tags[ith_cell])
                    elif geometry == 'wedge':
                        self.mat_cell_info_dict[cell_count] = \
                            Wedge(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                  np.array([self.perm_x_cell[mat_count],
                                            self.perm_y_cell[mat_count],
                                            self.perm_z_cell[mat_count]]), tags[ith_cell])
                    elif geometry == 'tetra':
                        self.mat_cell_info_dict[cell_count] = \
                            Tetrahedron(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                        np.array([self.perm_x_cell[mat_count],
                                                  self.perm_y_cell[mat_count],
                                                  self.perm_z_cell[mat_count]]),
                                        tags[ith_cell])
                    elif geometry == 'pyramid':
                        self.mat_cell_info_dict[cell_count] = \
                            Pyramid(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                    np.array([self.perm_x_cell[mat_count],
                                              self.perm_y_cell[mat_count],
                                              self.perm_z_cell[mat_count]]), tags[ith_cell])
                    cell_count += 1
                    mat_count += 1
                # boundary cells
                elif tags[ith_cell] in self.physical_tags['boundary']:
                    for key in nodes_to_cell:
                        self.bound_cells_to_node.setdefault(key, [])
                        self.bound_cells_to_node[key].append(face_count)

                    if geometry == 'quad':
                        self.bound_cell_info_dict[face_count] = \
                            Quadrangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0,
                                       tags[ith_cell])
                        self.boundary_conditions[tags[ith_cell]]['cells'].append(face_count)
                    elif geometry == 'triangle':
                        self.bound_cell_info_dict[face_count] = \
                            Triangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0,
                                     tags[ith_cell])
                        self.boundary_conditions[tags[ith_cell]]['cells'].append(face_count)
                    face_count += 1
                    bound_count += 1
                # fracture cells
                elif tags[ith_cell] in self.physical_tags['fracture']:
                    for key in nodes_to_cell:
                        self.frac_cells_to_node.setdefault(key, [])
                        self.frac_cells_to_node[key].append(cell_count)

                    if geometry == 'quad':
                        self.frac_cell_info_dict[cell_count] = \
                            Quadrangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0, tags[ith_cell])
                    elif geometry == 'triangle':
                        self.frac_cell_info_dict[cell_count] = \
                            Triangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0, tags[ith_cell])
                    cell_count += 1
                # output faces
                elif tags[ith_cell] in self.physical_tags['output']:
                    if geometry == 'quad':
                        self.output_face_info_dict[output_count] = \
                            Quadrangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0, tags[ith_cell])
                    elif geometry == 'triangle':
                        self.output_face_info_dict[output_count] = \
                            Triangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry, 0.0, tags[ith_cell])
                    output_count += 1

        self.matrix_cell_count = mat_count
        self.bound_cell_count = bound_count
        self.fracture_cell_count = cell_count - mat_count
        self.output_face_count = output_count

        print('Time to load Mesh: {:f} [sec]'.format((time.time() - start_time_module)))
        print(self.mesh_data)

    def write_to_vtk(self, output_directory, property_array, cell_property, ith_step):
        """
        Class method which writes output of unstructured grid to VTK format
        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :return:
        """
        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Temporarily store mesh_data in copy:
        Mesh = meshio.read(self.mesh_file)

        # Allocate empty new cell_data_dict dictionary:
        cell_data_dict = dict()

        for ith_prop in range(len(cell_property)):
            cell_data_dict[cell_property[ith_prop]] = []
            left_bound = 0
            right_bound = 0
            for ith_geometry in self.mesh_data.cells_dict:
                left_bound = right_bound
                right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]
                cell_data_dict[cell_property[ith_prop]].append(list(property_array[left_bound:right_bound, ith_prop]))

        cell_data_dict['matrix_cell_bool'] = []
        left_bound = 0
        right_bound = 0
        for ith_geometry in self.mesh_data.cells_dict:
            left_bound = right_bound
            right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]

            if (ith_geometry in self.available_fracture_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.zeros(((right_bound - left_bound),))))

            elif (ith_geometry in self.available_matrix_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.ones(((right_bound - left_bound),))))

        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            # Each item in cell data must match the cells array
            cell_data=cell_data_dict)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)
        return 0

    def calc_cell_information(self, cache=0):
        """"
        Class method which calculates the geometrical properties of the grid
        """
        start_time_module = time.time()
        read_from_cache = False

        if cache:
            if os.path.isfile(self.mesh_file + '.cellInfoDict.cache'):
                print('Load cell information from cache...')
                with open(self.mesh_file + '.cellInfoDict.cache', 'rb') as handle:
                    self.cellInfoDict = pickle.load(handle)

                self.mat_cells_to_node = self.cellInfoDict['mat_cells_to_node']
                self.frac_cells_to_node = self.cellInfoDict['frac_cells_to_node']
                self.mat_cell_info_dict = self.cellInfoDict['mat_cell_info_dict']
                self.frac_cell_info_dict = self.cellInfoDict['frac_cell_info_dict']
                self.matrix_cell_count = self.cellInfoDict['matrix_cell_count']
                self.fracture_cell_count = self.cellInfoDict['fracture_cell_count']
                read_from_cache = True
                print('Time to read cell info: {:f} [sec]'.format((time.time() - start_time_module)))
                print('------------------------------------------------\n')

        if not read_from_cache:
            print('Start calculation cell information...')
            # Find cells belonging to particular node:
            actual_mat_cell_id = 0
            actual_frac_cell_id = 0
            global_count = 0

            for geometry in self.geometries_in_mesh_file:
                # Main loop over different existing geometries
                for ith_cell, nodes_to_cell in enumerate(self.mesh_data.cells_dict[geometry]):
                    global_count += 1

                    # Calculate general information for cell, based on geometry, nodes belong to cell and their coordinates:
                    if geometry in self.available_matrix_geometries:
                        actual_mat_cell_id = self.matrix_cell_count + ith_cell

                        for ith_node in range(len(nodes_to_cell)):
                            key = nodes_to_cell[ith_node]
                            self.mat_cells_to_node.setdefault(key, [])
                            self.mat_cells_to_node[key].append(actual_mat_cell_id)

                        if ith_cell == (len(self.mesh_data.cells_dict[geometry]) - 1):
                            self.matrix_cell_count = self.matrix_cell_count + ith_cell + 1

                    elif geometry in self.available_fracture_geometries:
                        actual_frac_cell_id = self.fracture_cell_count + ith_cell

                        for ith_node in range(len(nodes_to_cell)):
                            key = nodes_to_cell[ith_node]
                            self.frac_cells_to_node.setdefault(key, [])
                            self.frac_cells_to_node[key].append(actual_frac_cell_id)

                        if ith_cell == (len(self.mesh_data.cells_dict[geometry]) - 1):
                            self.fracture_cell_count = self.fracture_cell_count + ith_cell + 1

                    # Construct control volume object depending on geometry (class):
                    if geometry == 'hexahedron':
                        self.mat_cell_info_dict[actual_mat_cell_id] = \
                            Hexahedron(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                       np.array([self.perm_x_cell[actual_mat_cell_id],
                                                 self.perm_y_cell[actual_mat_cell_id],
                                                 self.perm_z_cell[actual_mat_cell_id]]))

                    elif geometry == 'wedge':
                        self.mat_cell_info_dict[actual_mat_cell_id] = \
                            Wedge(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                  np.array([self.perm_x_cell[actual_mat_cell_id],
                                            self.perm_y_cell[actual_mat_cell_id],
                                            self.perm_z_cell[actual_mat_cell_id]]))

                    elif geometry == 'tetra':
                        self.mat_cell_info_dict[actual_mat_cell_id] = \
                            Tetrahedron(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                        np.array([self.perm_x_cell[actual_mat_cell_id],
                                                  self.perm_y_cell[actual_mat_cell_id],
                                                  self.perm_z_cell[actual_mat_cell_id]]))

                    elif geometry == 'pyramid':
                        self.mat_cell_info_dict[actual_mat_cell_id] = \
                            Pyramid(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                    np.array([self.perm_x_cell[actual_mat_cell_id],
                                              self.perm_y_cell[actual_mat_cell_id],
                                              self.perm_z_cell[actual_mat_cell_id]]))

                    elif geometry == 'quad':
                        self.frac_cell_info_dict[actual_frac_cell_id] = \
                            Quadrangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                       self.fracture_aperture[actual_frac_cell_id])

                    elif geometry == 'triangle':
                        self.frac_cell_info_dict[actual_frac_cell_id] = \
                            Triangle(nodes_to_cell, self.mesh_data.points[nodes_to_cell, :], geometry,
                                     self.fracture_aperture[actual_frac_cell_id])

            print('Time to calculate cell info: {:f} [sec]'.format((time.time() - start_time_module)))
            print('------------------------------------------------\n')

        # Some fail safe, check if number of fracture cells and number of
        # matrix cells are the same as pre-assumed count, if exists:
        if not self.matrix_cell_count == self.mat_cells_tot:
            print('Matrix cells found after calc. cell info not same as pre-assumed matrix cell count!!!!!!')
        if not self.fracture_cell_count == self.frac_cells_tot:
            print('Matrix cells found after calc. cell info not same as pre-assumed matrix cell count!!!!!!')

        if cache and not read_from_cache:
            self.cellInfoDict = {}
            self.cellInfoDict['mat_cells_to_node'] = self.mat_cells_to_node
            self.cellInfoDict['frac_cells_to_node'] = self.frac_cells_to_node
            self.cellInfoDict['mat_cell_info_dict'] = self.mat_cell_info_dict
            self.cellInfoDict['frac_cell_info_dict'] = self.frac_cell_info_dict
            self.cellInfoDict['matrix_cell_count'] = self.matrix_cell_count
            self.cellInfoDict['fracture_cell_count'] = self.fracture_cell_count

            with open(self.mesh_file + '.cellInfoDict.cache', 'wb') as handle:
                pickle.dump(self.cellInfoDict, handle, protocol=4)
            print("Files have been read and cached.")
        return 0

    def write_volume_to_file(self, file_name):
        """
        Class method which loops over all the cells and writes the volume into a file (first frac, then mat)
        :return:
        """
        # Temporarily write volume.dat file:
        file = open(file_name, 'w')
        file.write("VOL\n")
        # Write volumes for fractures:
        for ith_cell in self.frac_cell_info_dict:
            file.write("%12.10f" % self.frac_cell_info_dict[ith_cell].volume)
            file.write("\n")
            if self.frac_cell_info_dict[ith_cell].volume < 10 ** (-15):
                print('NEGATIVE FRACTURE CELL VOLUME:')
                print(self.frac_cell_info_dict[ith_cell].coord_nodes_to_cell)
                print(self.frac_cell_info_dict[ith_cell].nodes_to_cell)

        for ith_cell in self.mat_cell_info_dict:
            file.write("%12.10f" % self.mat_cell_info_dict[ith_cell].volume)
            file.write("\n")
            if self.mat_cell_info_dict[ith_cell].volume < 10 ** (-15):
                print('NEGATIVE MATRIX CELL VOLUME:')
                print(self.mat_cell_info_dict[ith_cell].coord_nodes_to_cell)
                print(self.mat_cell_info_dict[ith_cell].nodes_to_cell)

        file.write("/")
        file.close()
        return 0

    def write_depth_to_file(self, file_name):
        """
        Class method which loops over all the cells and writes the volume into a file (first frac, then mat)
        :return:
        """
        # Temporarily write volume.dat file:
        file = open(file_name, 'w')
        file.write("DEPTH\n")
        # Write volumes for fractures:
        for ith_cell in self.frac_cell_info_dict:
            file.write("%12.10f" % self.frac_cell_info_dict[ith_cell].depth)
            file.write("\n")

        for ith_cell in self.mat_cell_info_dict:
            file.write("%12.10f" % self.mat_cell_info_dict[ith_cell].depth)
            file.write("\n")

        file.write("/")
        file.close()
        return 0

    def store_volume_all_cells(self):
        """
        Class method which loops over all the cells and stores the volume in single array (first frac, then mat)
        :return:
        """
        tot_cell_count = 0
        for ith_cell in self.frac_cell_info_dict:
            self.volume_all_cells[tot_cell_count] = self.frac_cell_info_dict[ith_cell].volume
            tot_cell_count += 1

        for ith_cell in self.mat_cell_info_dict:
            self.volume_all_cells[tot_cell_count] = self.mat_cell_info_dict[ith_cell].volume
            tot_cell_count += 1

        self.volume_all_cells = np.array(list(self.volume_all_cells.values()))
        return 0

    def store_centroid_all_cells(self):
        """
        Class method which loops over all the cells and stores the volume in single array (first frac, then mat)
        :return:
        """
        tot_cell_count = 0
        for ith_cell in self.frac_cell_info_dict:
            self.centroid_all_cells[tot_cell_count] = self.frac_cell_info_dict[ith_cell].centroid
            tot_cell_count += 1

        for ith_cell in self.mat_cell_info_dict:
            self.centroid_all_cells[tot_cell_count] = self.mat_cell_info_dict[ith_cell].centroid
            tot_cell_count += 1

        self.centroid_all_cells = np.array(list(self.centroid_all_cells.values()))
        return 0

    def store_depth_all_cells(self):
        """
        Class method which loops over all the cells and stores the depth in single array (first frac, then mat)
        :return:
        """
        tot_cell_count = 0
        for ith_cell in self.frac_cell_info_dict:
            self.depth_all_cells[tot_cell_count] = self.frac_cell_info_dict[ith_cell].depth
            tot_cell_count += 1

        for ith_cell in self.mat_cell_info_dict:
            self.depth_all_cells[tot_cell_count] = self.mat_cell_info_dict[ith_cell].depth
            tot_cell_count += 1

        for ith_cell in self.bound_cell_info_dict:
            self.depth_all_cells[tot_cell_count] = self.bound_cell_info_dict[ith_cell].depth
            tot_cell_count += 1

        self.depth_all_cells = np.array(list(self.depth_all_cells.values()))
        return 0

    @staticmethod
    def write_conn2p_to_file(cell_m, cell_p, tran, file_name):
        """
        Static method which write a connection list to the specified file (for non-thermal application)
        :param cell_m: negative residual contribution of cell block connections of interface
        :param cell_p: positive residual contribution of cell block connections of interface
        :param tran: transmissibility value of the interface
        :param file_name: file name where to write connection list
        :return:
        """
        file = open(file_name, 'w')
        file.write("TPFACONNS\n")
        file.write(str(cell_m.size) + "\n")
        for ith_conn in range(cell_m.size):
            file.write("%6d %6d %8.6f" % (cell_m[ith_conn], cell_p[ith_conn], tran[ith_conn]))
            file.write("\n")

        file.write("/")
        file.close()
        return 0

    @staticmethod
    def write_conn2p_therm_to_file(cell_m, cell_p, tran, tranD, file_name):
        """
        Static method which write a connection list to the specified file (for thermal application)
        :param cell_m: negative residual contribution of cell block connections of interface
        :param cell_p: positive residual contribution of cell block connections of interface
        :param tran: transmissibility value of the interface
        :param tranD: geometric coefficient of interface
        :param file_name: file name where to write connection list
        :return:
        """
        file = open(file_name, 'w')
        file.write("TPFACONNSN\n")
        file.write(str(cell_m.size) + "\n")
        for ith_conn in range(cell_m.size):
            file.write("%6d %6d %8.6f %8.6f" % (cell_m[ith_conn], cell_p[ith_conn], tran[ith_conn], tranD[ith_conn]))
            file.write("\n")

        file.write("/")
        file.close()
        return 0

    @staticmethod
    def write_property_to_file(data, key_word: str, file_name: str, num_cells: int):
        """
        Static method which writes any specified property (in data) to any specified file
        :param data: data object required to write to a file
        :param key_word: keyword (usually read by other simulator)
        :param file_name: name of the file where to write
        :param num_cells: number of reservoir blocks
        :return:
        """
        # Statis method which writes any property to a specified file:
        file = open(file_name, 'w')
        file.write("{:s}\n".format(key_word))
        for ith_cell in range(num_cells):
            if np.isscalar(data):
                file.write("%8.6f\n" % (data))
            else:
                file.write("%8.6f\n" % (data[ith_cell]))

        file.write("/")
        file.close()
        return 0

    def calc_boundary_cells_new(self, boundary_data):
        """
        Class method which calculates constant boundary values at a specif constant x,y,z-coordinate
        :param boundary_data: dictionary with the boundary direction (x,y, or z) and type (min or max)
        :return:
        """
        # Specify boundary cells, simply set specify the single coordinate which is not-changing and its value:
        index = []  # Dynamic list containing indices of the nodes (points) which lay on the boundary:
        if boundary_data['boundary_dir'] == 'X':
            # Check if coordinate of points is on the boundary:
            if boundary_data['boundary_type'] == 'min':
                index = self.mesh_data.points[:, 0] == np.min(self.mesh_data.points[:, 0])
            elif boundary_data['boundary_type'] == 'max':
                index = self.mesh_data.points[:, 0] == np.max(self.mesh_data.points[:, 0])

        elif boundary_data['boundary_dir'] == 'Y':
            # Check if coordinate of points is on the boundary:
            if boundary_data['boundary_type'] == 'min':
                index = self.mesh_data.points[:, 1] == np.min(self.mesh_data.points[:, 1])
            elif boundary_data['boundary_type'] == 'max':
                index = self.mesh_data.points[:, 1] == np.max(self.mesh_data.points[:, 1])

        elif boundary_data['boundary_dir'] == 'Z':
            # Check if coordinate of points is on the boundary:
            if boundary_data['boundary_type'] == 'min':
                index = self.mesh_data.points[:, 2] == np.min(self.mesh_data.points[:, 2])
            elif boundary_data['boundary_type'] == 'max':
                index = self.mesh_data.points[:, 2] == np.max(self.mesh_data.points[:, 2])

        # Convert dynamic list to numpy array:
        boundary_points = np.array(list(compress(range(len(index)), index)))

        # Find cells containing boundary cells, for wedges or hexahedrons, the boundary cells must contain,
        # on the X or Y boundary four nodes exactly and on the Z axis four or three ndoes!
        #     0------0          0
        #    /     / |         /  \
        #  0------0  0        0----0
        #  |      | /         |    |
        #  0------0           0----0
        # Hexahedron       Wedge (prism)
        # Create loop over all matrix cells which are of the geometry 'matrix_cell_type'
        count = 0  # Counter for number of matrix cells on the boundary
        boundary_cells = {}  # Dictionary with matrix cells on the boundary
        for geometry in self.geometries_in_mesh_file:
            if geometry in self.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 or 3 nodes which intersect
                # with the boundary_points list:
                for ith_cell, ith_row in enumerate(self.mesh_data.cells_dict[geometry]):
                    if len(set.intersection(set(ith_row), set(boundary_points))) == 4 or \
                            len(set.intersection(set(ith_row), set(boundary_points))) == 3:
                        # Store cell since it is on the left boundary:
                        boundary_cells[count] = ith_cell
                        count += 1

        boundary_cells = np.array(list(boundary_cells.values()), dtype=int) + self.fracture_cell_count
        return boundary_cells

    def calc_boundary_cells(self, boundary_data):
        """
        Class method which calculates constant boundary values at a specif constant x,y,z-coordinate
        :param boundary_data: dictionary with the boundary location (X,Y,Z, and location)
        :return:
        """
        # Specify boundary cells, simply set specify the single coordinate which is not-changing and its value:
        # First boundary:
        index = []  # Dynamic list containing indices of the nodes (points) which lay on the boundary:
        if boundary_data['first_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 0] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 1] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 2] == boundary_data['first_boundary_val']

        # Convert dynamic list to numpy array:
        left_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Second boundary (same as above):
        index = []
        if boundary_data['second_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 0] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 1] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.mesh_data.points[:, 2] == boundary_data['second_boundary_val']

        right_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Find cells containing boundary cells, for wedges or hexahedrons, the boundary cells must contain,
        # on the X or Y boundary four nodes exactly!
        #     0------0          0
        #    /     / |         /  \
        #  0------0  0        0----0
        #  |      | /         |    |
        #  0------0           0----0
        # Hexahedron       Wedge (prism)
        # Create loop over all matrix cells which are of the geometry 'matrix_cell_type'
        left_count = 0  # Counter for number of left matrix cells on the boundary
        left_boundary_cells = {}  # Dictionary with matrix cells on the left boundary
        for geometry in self.geometries_in_mesh_file:
            if geometry in self.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the left_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.mesh_data.cells_dict[geometry]):

                    if len(set.intersection(set(ith_row), set(left_boundary_points))) == 4 or \
                            len(set.intersection(set(ith_row), set(
                                left_boundary_points))) == 3:  # Store cell since it is on the left boundary:
                        left_boundary_cells[left_count] = ith_cell
                        left_count += 1

        right_count = 0
        right_boundary_cells = {}
        for geometry in self.geometries_in_mesh_file:
            if geometry in self.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the right_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.mesh_data.cells_dict[geometry]):
                    if len(set.intersection(set(ith_row), set(right_boundary_points))) == 4 or \
                            len(set.intersection(set(ith_row), set(
                                right_boundary_points))) == 3:  # Store cell since it is on the left boundary:
                        # Store cell since it is on the left boundary:
                        right_boundary_cells[right_count] = ith_cell
                        right_count += 1

        left_boundary_cells = np.array(list(left_boundary_cells.values()), dtype=int) + \
                              self.fracture_cell_count
        right_boundary_cells = np.array(list(right_boundary_cells.values()), dtype=int) + \
                               self.fracture_cell_count
        return left_boundary_cells, right_boundary_cells

    # Two-Point Flux Approximation (TPFA)
    def calc_connections_all_cells(self, cache=0):
        """
        Class methods which calculates the connection list for all cell types (matrix & fracture)
        :return cell_m: minus-side of the connection
        :return cell_p: plus-side of the connection
        :return tran: transmissibility value of connection
        :return tran_thermal: geometric coefficient of connection
        """
        # Method for fracture-fracture connections is partially analogues to
        # the matrix-matrix and matrix-fracture procedure(!)
        # Loop over all fracture cells:
        start_time_module = time.time()

        if cache:
            file_paths = [self.mesh_file + '.cell_m.cache', self.mesh_file + '.cell_p.cache', self.mesh_file + '.tran.cache',
                          self.mesh_file + '.tran_thermal.cache', self.mesh_file + '.conn_stats.cache']
            file_exists = []
            for ii in file_paths:
                file_exists.append(os.path.isfile(ii))

            if all(file_exists):
                # Only read if all five files required are cached:
                print('Read connection list from cache...')
                cell_m = np.fromfile(self.mesh_file + '.cell_m.cache', dtype=np.int32)
                cell_p = np.fromfile(self.mesh_file + '.cell_p.cache', dtype=np.int32)
                tran = np.fromfile(self.mesh_file + '.tran.cache')
                tran_thermal = np.fromfile(self.mesh_file + '.tran_thermal.cache')
                connection_stats = np.fromfile(self.mesh_file + '.conn_stats.cache', dtype=np.int32)
                print('Time to load connection list: {:f} [sec]'.format((time.time() - start_time_module)))
                print('\t#Frac-Frac connections found: {:d}'.format(connection_stats[0]))
                print('\t#Mat-Mat connections found:   {:d}'.format(connection_stats[1]))
                print('\t#Mat-Frac connections found:  {:d}'.format(connection_stats[2]))
                print('------------------------------------------------\n')
                return cell_m, cell_p, tran, tran_thermal

        print('Start calculation connection list for fracture-fracture (if present) connections...')
        count_connection = 0
        count_mat_mat_conn = 0
        count_mat_frac_conn = 0
        count_frac_frac_conn = 0
        offset_frac_cell_count = 0

        if self.fracture_cell_count == 0:
            offset_mat_cell_count = 0
        else:
            offset_mat_cell_count = self.fracture_cell_count

        # Back to using dictionaries, performance increase of ~700%
        cell_m = {}
        cell_p = {}
        tran = {}
        tran_thermal = {}

        for ith_frac, dummy in enumerate(self.frac_cell_info_dict):
            # Loop over all faces of fracture cell and determine intersections based on
            # nodes belonging to fracture (inter)face:
            for key, nodes_to_face in self.frac_cell_info_dict[ith_frac].nodes_to_faces.items():

                # Size of nodesToFace in 3D == 2, in 2D == 1.  This is because fractures
                # in 2D intersect in a point (which is one node) and
                # in 3D intersect in a line (which has two nodes)!

                # Determine number of connected fractures at node:
                # Fractures in 3D modelling domain always appear as 2D objects in GMSH
                # and 2D objects in a 3D space intersect in a line(!), therefore
                # require two common nodes to form an intersection:
                intsect_cells_of_face = list(set.intersection(set(self.frac_cells_to_node[nodes_to_face[0]]),
                                                              set(self.frac_cells_to_node[nodes_to_face[1]])))
                intsect_cells_of_face = np.sort(np.array(intsect_cells_of_face))
                num_connecting_frac = len(intsect_cells_of_face)

                if (num_connecting_frac > 1):
                    # At least two fracture cells are connected through this node:
                    # Compute star-delta transformation and store connectivity for
                    # for each unique fracture-fracture connection:
                    if intsect_cells_of_face[0] == ith_frac:

                        # If statement is necessary to have unique connections, since
                        # the main outer loop is over all fracture cells, and e.g. if
                        # fracture 5 and 6 are connected, would otherwise appear twice
                        # in the connection list (as (5, 6, Trans56) and (6, 5, Trans65))
                        local_frac_elem = np.array(intsect_cells_of_face)

                        # Find unique set of combinations for fracture connections
                        # Example (for 2D), Star-Delta transform:
                        #      1                1
                        #      |               /|\
                        #      |              / | \
                        # 0----x----2  -->   0--+--2
                        #      |              \ | /
                        #      |               \|/
                        #      3                3
                        # Means 3*2*1 = 6 possible combinations of fracture
                        # connections. Generally: (N-1)! combinations for N is
                        # number of fracture intersections
                        for ith_connection in combinations(local_frac_elem, 2):
                            connect_array = np.array(ith_connection)

                            # Extract temporary fracture element (for which the connection is calculated)
                            temp_frac_elem = local_frac_elem[local_frac_elem != connect_array[0]]

                            # FUNCTION TO COMPUTE ALPHA AND STAR DELTA HERE
                            trans_i_j, thermal_i_j = \
                                TransCalculations.calc_trans_frac_frac(connect_array, temp_frac_elem,
                                                                       self.frac_cell_info_dict,
                                                                       self.mesh_data.points[nodes_to_face])

                            # Instead of appending, use list or dictionary:
                            cell_m[count_connection] = connect_array[0] + offset_frac_cell_count
                            cell_p[count_connection] = connect_array[1] + offset_frac_cell_count
                            tran[count_connection] = trans_i_j
                            tran_thermal[count_connection] = thermal_i_j

                            # Update counters:
                            count_connection += 1  # Total connections counter
                            count_frac_frac_conn += 1  # Fracture-Fracture connections counter

        print('Time to calculate connection list: {:f} [sec]'.format((time.time() - start_time_module)))
        print('\t#Frac-Frac connections found: {:d}'.format(count_frac_frac_conn))
        print('------------------------------------------------\n')

        # Start code for Matrix-Matrix and Fracture-Matrix connections:
        start_time_module = time.time()
        print('Start calculation connection list for matrix-matrix and matrix-fracture (if present) connections...')
        # Loop over all matrix-matrix connection and incidental matrix-fracture connections:
        for ith_cell, dummy in enumerate(self.mat_cell_info_dict):
            for key, nodes_to_face in self.mat_cell_info_dict[ith_cell].nodes_to_faces.items():  # for every face
                # nodeToFace is nodes belonging to face
                # cellsToNode is cells belonging to node
                # cellsToNode[nodeToFace[ii]] is cells belonging to ii-th node on face
                # so if {cellsToNode[nodeToFace[ii]]}, where ii is an element of all the
                # facial nodes, has an intersection of exactly two cells,
                # then those two cells must be neighbors and share this particular interface
                face_has_fracture = False

                # Find intersections of matrix cells for particular face (depending on
                # the geometry of the matrix cell):
                intsect_cells_to_face = self.mat_cell_info_dict[ith_cell].find_intersections(self.mat_cells_to_node,
                                                                                             nodes_to_face)

                # Since an interface, in whatever dimension, requires to adjacent cells
                # in order to form a valid entry for our connection list, start the
                # transmissibility calculations on when connection has been found:
                if len(intsect_cells_to_face) is 2:
                    intsect_cells_to_face = sorted(list(intsect_cells_to_face))

                    # If statement is necessary to have unique connections, since
                    # the main outer loop is over all matrix cells, and e.g. if
                    # matrix 5 and 6 are connected, would otherwise appear twice
                    # in the connection list (as (5, 6, Trans56) and (6, 5, Trans65)):
                    if intsect_cells_to_face[0] == ith_cell:

                        # Check if any fractures are present throughout the domain:
                        if self.fracture_cell_count > 0:

                            # Check if there exists a fracture on the currently investigated
                            # interface:
                            try:
                                for geometry in self.frac_geometries_in_file:
                                    # todo: this is most likely the reason why models with a lot of fractures are
                                    #  extremely slow, check after holiday if this can be done in another way!!!
                                    if any(np.equal(np.sort(self.mesh_data.cells_dict[geometry], axis=1),
                                                    np.sort(nodes_to_face)).all(1)):
                                        '''
                                        I am sorting here because I checking the nodes of the face belonging to the cell,
                                        which can be in any order, versus the nodes belonging to all the fracture cell
                                        geometries. If any match exists, then there must be a fracture on the face.
                                        '''
                                        face_has_fracture = True

                                        # Find intersection fracture (fracture element number):
                                        frac_element_nr = list(
                                            self.mat_cell_info_dict[ith_cell].find_intersections(self.frac_cells_to_node,
                                                                                                 nodes_to_face))

                            except ValueError:
                                pass

                        # Information on possible fracture is known and immediately used:
                        if face_has_fracture:
                            # Calculate transmissibility between fracture and matrix
                            # for cell_i:
                            trans_i_j, thermal_i_j = \
                                TransCalculations.calc_trans_mat_frac(intsect_cells_to_face[0],
                                                                      frac_element_nr[0],
                                                                      self.mat_cell_info_dict,
                                                                      self.frac_cell_info_dict,
                                                                      self.mesh_data.points[nodes_to_face, :])

                            # Instead of appending, use list or dictionary:
                            cell_m[count_connection] = intsect_cells_to_face[0] + offset_mat_cell_count
                            cell_p[count_connection] = frac_element_nr[0] + offset_frac_cell_count
                            tran[count_connection] = trans_i_j
                            tran_thermal[count_connection] = thermal_i_j

                            # Update counters:
                            count_connection += 1  # Total connections counter
                            count_mat_frac_conn += 1  # Matrix-Fracture connections counter

                            # Correct volume of cell_i for presence of fracture:
                            if 1 / 2 * self.frac_cell_info_dict[frac_element_nr[0]].volume >= \
                                    self.mat_cell_info_dict[intsect_cells_to_face[0]].volume:
                                print('Found very small matrix element: {:d}'.format(intsect_cells_to_face[0]))
                                print('Correcting for fracture volume results in negative volume')
                                self.mat_cell_info_dict[intsect_cells_to_face[0]].volume = \
                                    self.frac_cell_info_dict[frac_element_nr[0]].volume
                            else:
                                self.mat_cell_info_dict[intsect_cells_to_face[0]].volume += \
                                    -1 / 2 * self.frac_cell_info_dict[frac_element_nr[0]].volume

                            # Calculate transmissibility between fracture and matrix
                            # for cell_j:
                            trans_i_j, thermal_i_j = \
                                TransCalculations.calc_trans_mat_frac(intsect_cells_to_face[1],
                                                                      frac_element_nr[0],
                                                                      self.mat_cell_info_dict,
                                                                      self.frac_cell_info_dict,
                                                                      self.mesh_data.points[nodes_to_face, :])

                            cell_m[count_connection] = intsect_cells_to_face[1] + offset_mat_cell_count
                            cell_p[count_connection] = frac_element_nr[0] + offset_frac_cell_count
                            tran[count_connection] = trans_i_j
                            tran_thermal[count_connection] = thermal_i_j

                            # Update counters:
                            count_connection += 1  # Total connections counter
                            count_mat_frac_conn += 1  # Matrix-Fracture connections counter

                            # Correct volume of cell_j for presence of fracture:
                            # Correct volume of cell_i for presence of fracture:
                            if 1 / 2 * self.frac_cell_info_dict[frac_element_nr[0]].volume >= \
                                    self.mat_cell_info_dict[intsect_cells_to_face[1]].volume:
                                print('Found very small matrix element: {:d}'.format(intsect_cells_to_face[1]))
                                print('Correcting for fracture volume results in negative volume')
                                self.mat_cell_info_dict[intsect_cells_to_face[1]].volume = \
                                    self.frac_cell_info_dict[frac_element_nr[0]].volume
                            else:
                                self.mat_cell_info_dict[intsect_cells_to_face[1]].volume += \
                                    -1 / 2 * self.frac_cell_info_dict[frac_element_nr[0]].volume

                        else:
                            # Calculate matrix-matrix transmissibility:
                            trans_i_j, thermal_i_j = \
                                TransCalculations.calc_trans_mat_mat(intsect_cells_to_face,
                                                                     self.mat_cell_info_dict,
                                                                     self.mesh_data.points[nodes_to_face, :])

                            cell_m[count_connection] = intsect_cells_to_face[0] + offset_mat_cell_count
                            cell_p[count_connection] = intsect_cells_to_face[1] + offset_mat_cell_count
                            tran[count_connection] = trans_i_j
                            tran_thermal[count_connection] = thermal_i_j

                            # Update counters:
                            count_connection += 1  # Total connections counter
                            count_mat_mat_conn += 1  # Matrix-Matrix connections counter

        print('Time to calculate connection list: {:f} [sec]'.format((time.time() - start_time_module)))
        print('\t#Mat-Mat connections found:   {:d}'.format(count_mat_mat_conn))
        print('\t#Mat-Frac connections found:  {:d}'.format(count_mat_frac_conn))
        print('------------------------------------------------\n')

        # Convert dictionary back to numpy array (try to find more optimized method...):
        cell_m = np.array(list(cell_m.values()), dtype=np.int32)
        cell_p = np.array(list(cell_p.values()), dtype=np.int32)
        tran = np.array(list(tran.values()))
        tran_thermal = np.array(list(tran_thermal.values()))

        if cache:
            # if caching is enabled, save to cache file
            cell_m.tofile(self.mesh_file + '.cell_m.cache')
            cell_p.tofile(self.mesh_file + '.cell_p.cache')
            tran.tofile(self.mesh_file + '.tran.cache')
            tran_thermal.tofile(self.mesh_file + '.tran_thermal.cache')
            connection_stats = np.array([count_frac_frac_conn, count_mat_mat_conn, count_mat_frac_conn])
            connection_stats.tofile(self.mesh_file + '.conn_stats.cache')
            print("Connection list has been constructed and cached.")

        return cell_m, cell_p, tran, tran_thermal

    # Multi-Point Flux Approximation (MPFA)
    def calc_cell_neighbours(self):
        output_faces = {frozenset(face.nodes_to_cell): face_id for face_id, face in self.output_face_info_dict.items()}

        self.faces = {}
        self.output_face_to_face = {}
        # matrix-matrix & matrix-boundary connections
        for node_num, cells_num in self.mat_cells_to_node.items():
            # All the cells' faces belonging to the interaction region
            cell_faces = {(i, j): face for i in cells_num for j, face in self.mat_cell_info_dict[i].nodes_to_faces.items() if
                          node_num in face}

            for id, pts in cell_faces.items():
                if id[0] not in self.faces: self.faces[id[0]] = {}
                if id[1] not in self.faces[id[0]]:
                    for id1, pts1 in cell_faces.items():
                        if id != id1 and np.all(np.in1d(pts, pts1)):
                            self.faces[id[0]][id[1]] = Face(id[0],id[1],id1[0],id1[1],self.mesh_data.points[pts],0,FType.MAT)
                            frozenpts = frozenset(pts)
                            if frozenpts in output_faces:
                                self.output_face_to_face[output_faces[frozenpts]] = (id[0], id[1])
                            if id1[0] not in self.faces: self.faces[id1[0]] = {}
                            self.faces[id1[0]][id1[1]] = Face(id1[0],id1[1],id[0],id[1],self.mesh_data.points[pts],0,FType.MAT)
                    if id[1] not in self.faces[id[0]]:
                        b_cells = [self.bound_cells_to_node[pt] for pt in pts]
                        b_id = next(iter(set(b_cells[0]).intersection(*b_cells)))
                        self.faces[id[0]][id[1]] = Face(id[0],id[1],id[0],b_id,self.mesh_data.points[pts],0,FType.BORDER)
                        frozenpts = frozenset(pts)
                        if frozenpts in output_faces:
                            self.output_face_to_face[output_faces[frozenpts]] = (id[0], id[1])

        fap = self.fracture_aperture
        # fracture-fracture connections
        for node_num, cells_num in self.frac_cells_to_node.items():
            frac_cell_faces = {(i, j): face for i in cells_num for j, face in self.frac_cell_info_dict[i].nodes_to_faces.items() if
                          node_num in face}
            if len(cells_num) > 1 :
                for id, pts in frac_cell_faces.items():
                    if id[0] not in self.faces: self.faces[id[0]] = {}
                    if id[1] not in self.faces[id[0]]:
                        for id1, pts1 in frac_cell_faces.items():
                            if id != id1 and np.all(np.in1d(pts, pts1)):
                                n = self.frac_cell_info_dict[id[0]].centroid - self.frac_cell_info_dict[id1[0]].centroid
                                self.faces[id[0]][id[1]] = Face(id[0], id[1], id1[0], id1[1], self.mesh_data.points[pts],
                                                                0, FType.FRAC, fap, n)
                                if id1[0] not in self.faces: self.faces[id1[0]] = {}
                                self.faces[id1[0]][id1[1]] = Face(id1[0], id1[1], id[0], id[1], self.mesh_data.points[pts],
                                                                  0, FType.FRAC, fap, n)
        # fracture-matrix connections
        for frac_id, cell in self.frac_cell_info_dict.items():
            mat_cell_faces = {(i, j): face for i in self.mat_cells_to_node[cell.nodes_to_cell[0]] for j, face in self.mat_cell_info_dict[i].nodes_to_faces.items() if
                          cell.nodes_to_cell[0] in face}
            counter = 4
            for id, pts in mat_cell_faces.items():
                if np.all(np.in1d(pts, cell.nodes_to_cell)):
                    #detached = self.faces[id[0]][id[1]]
                    #detached.cell_id2 = frac_id
                    #detached.face_id2 = counter
                    #detached.type = FType.MAT_TO_FRAC
                    cur_mat = self.faces[id[0]]
                    cur_mat[len(cur_mat)] = Face(id[0], id[1], frac_id, counter, self.mesh_data.points[pts],
                                                                  0, FType.MAT_TO_FRAC)
                    self.faces[frac_id][counter] = Face(frac_id, counter, id[0], id[1], self.mesh_data.points[pts],
                                                                  0, FType.FRAC_TO_MAT)
                    t_face = cur_mat[len(cur_mat)-1].centroid - self.mat_cell_info_dict[id[0]].centroid
                    cur_mat[len(cur_mat) - 1].n *= 1.0 if cur_mat[len(cur_mat) - 1].n.dot(t_face) < 0 else -1.0
                    self.faces[frac_id][counter].n *= 1.0 if self.faces[frac_id][counter].n.dot(t_face) < 0 else -1.0
                    counter += 1
    def add_subinterface(self, src, src_cells):
        idx1 = np.abs(src) > self.tol
        idx2 = np.argsort(-np.abs(src[idx1]))
        #assert(len(main_cells) < 2 or main_cells == {src_cells[idx1][idx2[0]], src_cells[idx1][idx2[1]]})
        return np.array(src_cells, dtype=np.int32)[idx1][idx2], src[idx1][idx2]
    def merge_subinterfaces(self, dest, key, src, src_cells, key1):
        assert( (key[0] == key1[0] and key[1] == key1[1]) or
                (key[0] == key1[1] and key[1] == key1[0]))
        mult = 1.0 if key[0] == key1[0] else -1.0
        idx = np.abs(src) > self.tol
        cells = np.array(src_cells, dtype=np.int32)[idx]
        trans = src[idx]
        for i in range(cells.size):
            idx = np.where(dest[0] == cells[i])
            if idx[0].size > 0:
                dest[1][idx[0][0]] += mult * trans[i]
            else:
                dest = (np.append(dest[0], cells[i]), np.append(dest[1], mult * trans[i]))

        #assert(len(main_cells) < 2 or main_cells == {dest[0][0], dest[0][1]})
        return dest
    def write_mpfa_conn_to_file(self, path = 'mpfa_conn.dat'):
        f = open(path, 'w')
        f.write(str(self.mpfa_connections_num) + '\n')
        for ids, data in self.mpfa_connections.items():
            cells = list(ids)
            isBound = cells[0][0] == cells[1][0]
            if not isBound:
                row = str(cells[0][0]) + '\t' + str(cells[1][0]) + '\t'
            else:
                row = str(cells[0][0]) + '\t' + str(self.mat_cells_tot + self.connections[cells[0][0]][cells[0][1]][1]) + '\t'
            for i in range(data[0].size):  row += '\t' + str(data[0][i]) + '\t' + str('{:.2e}'.format(data[1][i]))
            f.write(row + '\n')
        f.close()
    def calc_mpfa_connections_all_cells(self, make_connections=False):
        # print('Start calculation connection list for fracture-fracture (if present) connections...')
        # Start code for Matrix-Matrix and Fracture-Matrix connections:
        start_time_module = time.time()
        print('Start calculation MPFA connection list for matrix-matrix and matrix-fracture (if present) connections...')

        trans_per_regions = {}
        faces_per_regions = {}
        # Loop over matrix nodes (one interaction regions per node)
        if make_connections:
            self.connections = {}
        for node_num, cells_num in self.mat_cells_to_node.items():
            # Cells in the interaction region
            cells = {id: self.mat_cell_info_dict[id] for id in cells_num}
            # All the cells' faces belonging to the interaction region
            cell_faces = {(i, j): face for i, cell in cells.items() for j, face in cell.nodes_to_faces.items() if
                          node_num in face}
            # Original faces - their positions (cell id, nodes_to_faces id) in all cells
            faces = {}
            i = 0
            # Number of boundary faces in the interaction region
            bc_faces_num = 0
            for id, item in cell_faces.items():
                isFound = False
                for id1, item1 in cell_faces.items():
                    if id != id1 and np.all(np.in1d(item, item1)):
                        isFound = True
                        if (id1, id) not in faces.values():
                            faces[i] = (id, id1)
                            i += 1
                        break
                if not isFound:
                    faces[i] = (id, id)
                    i += 1
                    bc_faces_num += 1
            # Dimensions of the interaction region

            n_faces = len(faces)
            # Map cell id to n_dim faces belonging to it in the interaction region
            cell_to_faces = {}
            for i, id_cell in enumerate(cells.keys()):
                tri_faces = [id_face for id_face, face in faces.items() if
                             id_cell == face[0][0] or id_cell == face[1][0]]
                cell_to_faces[i] = tri_faces
            # Faces' centers
            face_centers = np.empty((n_faces, self.n_dim), dtype=np.float64)
            for id, face in faces.items():
                face_centers[id] = np.average(self.mesh_data.points[cells[face[0][0]].nodes_to_faces[face[0][1]], :],
                                              axis=0)

            n_cells = len(cells) + bc_faces_num
            cur_stencil = -np.ones(n_cells, dtype=np.int)
            cur_stencil[:len(cells)] = cells_num
            nu = np.zeros((self.n_dim, self.n_dim), dtype=np.float64)
            omega = np.zeros(self.n_dim, dtype=np.float64)
            A = np.zeros((n_faces, n_faces), dtype=np.float64)
            B = np.zeros((n_faces, n_cells), dtype=np.float64)
            C = np.zeros((n_faces, n_faces), dtype=np.float64)
            D = np.zeros((n_faces, n_cells), dtype=np.float64)
            # Basis construction of dual basis for each cell and assembling of matrices
            bc_counter = len(cells)
            # Inner subinterfaces & Dirichlet boundaries
            inds_to_check = [*range(n_cells)]
            pt0 = self.mesh_data.points[node_num]
            for i, id_cell in enumerate(cells.keys()):
                id_faces = cell_to_faces[i]
                t_faces = face_centers[id_faces] - cells[id_cell].centroid[None, :]

                nu[0] = np.cross(t_faces[1], t_faces[2])
                nu[1] = np.cross(t_faces[0], t_faces[2])
                nu[2] = np.cross(t_faces[0], t_faces[1])
                if np.inner(t_faces[0], nu[0]) < 0: nu[0] = -nu[0]
                if np.inner(t_faces[1], nu[1]) < 0: nu[1] = -nu[1]
                if np.inner(t_faces[2], nu[2]) < 0: nu[2] = -nu[2]
                vol = np.inner(t_faces[0], nu[0])
                assert(vol > 0)
                all_pts = [set(cells[faces[id_faces[k]][0][0]].nodes_to_faces[faces[id_faces[k]][0][1]]) for k in
                       range(self.n_dim)]
                c_pts = [set.intersection(all_pts[1], all_pts[2]), set.intersection(all_pts[0], all_pts[2]), set.intersection(all_pts[0], all_pts[1])]
                ad_pts = [list(c_pts[1].symmetric_difference(c_pts[2])), list(c_pts[0].symmetric_difference(c_pts[2])),
                       list(c_pts[0].symmetric_difference(c_pts[1]))]
                # Iterations over n_dim faces bounding the cell in the interaction region
                for k in range(self.n_dim):
                    edges = [(self.mesh_data.points[ad_pts[k][0]] - pt0) / 2,
                             face_centers[id_faces[k]] - pt0,
                             (self.mesh_data.points[ad_pts[k][1]] - pt0) / 2]
                    n = (np.cross(edges[0], edges[1]) + np.cross(edges[1], edges[2])) / 2
                    dir = face_centers[id_faces[k]] - self.mat_cell_info_dict[faces[id_faces[k]][0][0]].centroid
                    if np.inner(dir, n) < 0: n = -n
                    # One side of flux continuity over id_faces[k] sub-interface
                    omega = -nu.dot(n.dot(self.permeability[id_cell])) / vol
                    if np.count_nonzero(C[id_faces[k]]) == 0:
                        C[id_faces[k], id_faces] -= TransCalculations.darcy_constant * omega
                        D[id_faces[k], i] -= TransCalculations.darcy_constant * np.sum(omega)
                    if faces[id_faces[k]][0][0] != faces[id_faces[k]][1][0]:
                        # Inner subinterface
                        mult = 1.0 if id_cell == faces[id_faces[k]][0][0] else -1.0
                        A[id_faces[k], id_faces] += mult * omega
                        B[id_faces[k], i] += mult * np.sum(omega)

                        if make_connections:
                            # First side
                            if faces[id_faces[k]][0][0] not in self.connections:
                                self.connections[faces[id_faces[k]][0][0]] = {}
                            if faces[id_faces[k]][0][1] not in self.connections[faces[id_faces[k]][0][0]]:
                                self.connections[faces[id_faces[k]][0][0]][faces[id_faces[k]][0][1]] = faces[id_faces[k]][1]
                            # Second side
                            if faces[id_faces[k]][1][0] not in self.connections:
                                self.connections[faces[id_faces[k]][1][0]] = {}
                            if faces[id_faces[k]][1][1] not in self.connections[faces[id_faces[k]][1][0]]:
                                self.connections[faces[id_faces[k]][1][0]][faces[id_faces[k]][1][1]] = faces[id_faces[k]][0]
                    else:
                        # Boundary subinterface
                        # Estimated boundary cell's id
                        cur_pts = cells[faces[id_faces[k]][0][0]].nodes_to_faces[faces[id_faces[k]][0][1]]
                        b_cells = [self.bound_cells_to_node[pt] for pt in cur_pts]
                        b_id = next(iter(set(b_cells[0]).intersection(*b_cells)))
                        if make_connections:
                            # One side
                            if faces[id_faces[k]][0][0] not in self.connections:
                                self.connections[faces[id_faces[k]][0][0]] = {}
                            if faces[id_faces[k]][0][1] not in self.connections[faces[id_faces[k]][0][0]]:
                                self.connections[faces[id_faces[k]][0][0]][faces[id_faces[k]][0][1]] = (faces[id_faces[k]][0][0], b_id)

                        bc_data = self.boundary_conditions[self.bound_cell_info_dict[b_id].prop_id]['flow']
                        assert(abs(bc_data['a']) + abs(bc_data['b']) > 0)
                        if bc_data['b'] != 0.0:
                            inds_to_check.remove(bc_counter)
                            # One side
                            A[id_faces[k], id_faces] += omega
                            B[id_faces[k], i] += np.sum(omega)
                            # Second side
                            A[id_faces[k], id_faces[k]] += bc_data['a'] / bc_data['b']
                            B[id_faces[k], bc_counter] += 1.0 / bc_data['b'] / len(cur_pts)
                        else:
                            A[id_faces[k], id_faces[k]] = 1.0
                            B[id_faces[k], bc_counter] = 1.0 / bc_data['a']

                        cur_stencil[bc_counter] = self.mat_cells_tot + b_id
                        bc_counter += 1
            # Transmissibility of sub-interface
            T = D - C.dot(np.linalg.inv(A)).dot(B)
            # Check if the sum of transmissibilities is not equal to zero
            sum_trans = np.abs(np.sum(T[:,inds_to_check], axis=1))
            assert (sum_trans < self.tol).all()
            # Store transmissibilities of all sub-interfaces and corresponding cell ids
            trans_per_regions[node_num] = T
            faces_per_regions[node_num] = (faces, cur_stencil)
        # Merge sub-interfaces and store connections
        self.mpfa_connections = {}
        self.mpfa_connections_num = 0
        for node_num, cells_num in self.mat_cells_to_node.items():
            for i, face in faces_per_regions[node_num][0].items():
                key1 = (face[0], face[1])
                key2 = (face[1], face[0])
                if key1 not in self.mpfa_connections:
                    c, trans = self.add_subinterface(trans_per_regions[node_num][i], faces_per_regions[node_num][1])
                    self.mpfa_connections[key1] = (c, trans)
                    if face[0][0] != face[1][0]:
                        self.mpfa_connections[key2] = (c, -trans)
                else:
                    # Merge sub-interfaces
                    self.mpfa_connections[key1] = self.merge_subinterfaces(self.mpfa_connections[key1],
                                                            (key1[0][0], key1[1][0]),
                                                            trans_per_regions[node_num][i],
                                                            faces_per_regions[node_num][1],
                                                            (face[0][0], face[1][0]) )
                    if face[0][0] != face[1][0]:
                        self.mpfa_connections[key2] = self.merge_subinterfaces(self.mpfa_connections[key2],
                                                            (key2[0][0], key2[1][0]),
                                                            trans_per_regions[node_num][i],
                                                            faces_per_regions[node_num][1],
                                                            (face[0][0], face[1][0]) )

        self.mpfa_connections_num = len(self.mpfa_connections)
        cell_m = []
        cell_p = []
        stencil = []
        offset = []
        trans = []
        accum_size = 0
        for ids, data in self.mpfa_connections.items():
            cells = list(ids)
            isBound = cells[0][0] == cells[1][0]
            cell_m.append(cells[0][0])
            if not isBound:
                cell_p.append(cells[1][0])
            else:
                cell_p.append(self.mat_cells_tot + self.connections[cells[0][0]][cells[0][1]][1])
            offset.append(accum_size)
            st_size = len(data[0])
            accum_size += st_size
            stencil.extend(data[0])
            trans.extend(data[1])
        offset.append(accum_size)

        print('Time to calculate MPFA connection list: {:f} [sec]'.format((time.time() - start_time_module)))
        print('\t#Mat-Mat MPFA connections found:   {:d}'.format(self.mpfa_connections_num))
        #print('\t#Mat-Frac MPFA connections found:  {:d}'.format(count_mat_frac_conn))
        print('------------------------------------------------\n')

        return cell_m, cell_p, stencil, offset, trans
    # Multi-Point Stress Approximation (MPSA)
    def get_isotropic_stiffness(self, E, nu):
        la = nu * E / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        return np.array([[la + 2 * mu, la, la, 0, 0, 0],
                         [la, la + 2 * mu, la, 0, 0, 0],
                         [la, la, la + 2 * mu, 0, 0, 0],
                         [0, 0, 0, mu, 0, 0],
                         [0, 0, 0, 0, mu, 0],
                         [0, 0, 0, 0, 0, mu]])
    def get_stiffness_submatrices(self, s):
        return np.array([[[s[0, 0], s[0, 5], s[0, 4]],
                          [s[0, 5], s[5, 5], s[4, 5]],
                          [s[0, 4], s[4, 5], s[4, 4]]],
                         [[s[5, 5], s[1, 5], s[3, 5]],
                          [s[1, 5], s[1, 1], s[1, 3]],
                          [s[3, 5], s[1, 3], s[3, 3]]],
                         [[s[4, 4], s[3, 4], s[2, 4]],
                          [s[3, 4], s[3, 3], s[2, 3]],
                          [s[2, 4], s[2, 3], s[2, 2]]],
                         [[s[4, 5], s[1, 4], s[3, 4]],
                          [s[3, 5], s[1, 3], s[3, 3]],
                          [s[2, 5], s[1, 2], s[2, 3]]],
                         [[s[0, 4], s[4, 5], s[4, 4]],
                          [s[0, 3], s[3, 5], s[3, 4]],
                          [s[0, 2], s[2, 5], s[2, 4]]],
                         [[s[0, 5], s[5, 5], s[4, 5]],
                          [s[0, 1], s[1, 5], s[1, 4]],
                          [s[0, 3], s[3, 5], s[3, 4]]]])
    def get_product_decompostion(self, A, n):
        tmp = A.dot(n)
        alpha = tmp.dot(n)
        g = tmp - np.tile(n, (6, 1)) * alpha[:, np.newaxis]
        gT = np.transpose(A, [0, 2, 1]).dot(n) - np.tile(n, (6, 1)) * alpha[:, np.newaxis]
        return alpha, g, gT
    def get_normal_tangential_stiffness(self, A, n):
        alpha, g, gT = self.get_product_decompostion(A, n)
        return np.array([[alpha[0], alpha[5], alpha[4]],
                         [alpha[5], alpha[1], alpha[3]],
                         [alpha[4], alpha[3], alpha[2]]]), np.array([np.concatenate((g[0], g[5], g[4])),
                                                                     np.concatenate((gT[5], g[1], g[3])),
                                                                     np.concatenate((gT[4], gT[3], g[2]))])
    def check_if_neumann_boundary(self, cell_id, face_id):
        face = self.faces[cell_id][face_id]
        if face.type == FType.BORDER:
            bc_data = self.bc_mech[self.bcm_num * face.face_id2:self.bcm_num * (face.face_id2 + 1)]
            if bc_data[2] == self.P12:
                return bc_data[0] == 0.0
        return False
    def check_if_roller_boundary(self, cell_id, face_id):
        face = self.faces[cell_id][face_id]
        if face.type == FType.BORDER:
            bc_data = self.bc_mech[self.bcm_num * face.face_id2:self.bcm_num * (face.face_id2 + 1)]
            return bc_data[2] == self.Prol
        return False
    def check_if_boundary(self, cell_id, face_id):
        nebr = self.faces[cell_id][face_id].cell_id2
        return nebr == cell_id
    def get_not_coplanar_stencil(self, cell_id, face_id):
        cell = self.mat_cell_info_dict[cell_id]
        faces = self.faces[cell_id]
        face = faces[face_id]
        # Check if is_on_fault
        is_on_fault = False
        fault_conn_id = -1
        for j, f in faces.items():
            if f.type == FType.MAT_TO_FRAC:
                is_on_fault = True
                fault_conn_id = j
                break
        #j0 = np.argwhere(cur_conns == faces[fault_conn_id].face_id1)[0][0]
        #face0 = self.faces[cell_id][cur_conns[j0]]

        t_face = face.centroid - cell.centroid
        # Determine first face
        face_id1 = (face_id - 1) % len(cell.nodes_to_faces)
        while face_id1 != face_id:
            face1 = faces[face_id1]
            t_face1 = face1.centroid - cell.centroid
            S = np.linalg.norm(np.cross(t_face, t_face1))
            Scmp = (np.pi / 180) * min(np.linalg.norm(t_face),np.linalg.norm(t_face1)) ** 2
            # Check non-collinearity & if it is not a Neumann boundary
            if S > Scmp and S > self.tol and face1.type != FType.MAT_TO_FRAC and not self.check_if_roller_boundary(cell_id, face_id1):
                break
            face_id1 = (face_id1 - 1) % len(cell.nodes_to_faces)
        # Determine second face
        face_id2 = (face_id + 1) % len(cell.nodes_to_faces)
        while face_id2 != face_id1:
            face2 = faces[face_id2]
            t_face2 = face2.centroid - cell.centroid
            V = np.abs(np.linalg.det(np.array([t_face, t_face1, t_face2])))
            Vcmp = (np.pi / 180) ** 2 * min(np.linalg.norm(t_face),np.linalg.norm(t_face1),np.linalg.norm(t_face2)) ** 3
            # Check non-complanarity & if it is not a Neumann boundary
            if V > Vcmp and V > self.tol and face1.type != FType.MAT_TO_FRAC and not self.check_if_roller_boundary(cell_id, face_id2):
                break
            face_id2 = (face_id2 + 1) % len(cell.nodes_to_faces)
        assert (face_id1 != face_id2)
        assert (face_id1 != face_id)
        assert (face_id2 != face_id)

        interfaces = [face_id1, face_id, face_id2]
        connections = [face_id1, face_id, face_id2]
        if is_on_fault and faces[fault_conn_id].face_id1 in connections: connections.append(fault_conn_id)
        return np.array(connections, dtype=np.intp), np.array(interfaces, dtype=np.intp)
    def get_full_stencil(self, cell_id, face_id=0):
        cell = self.mat_cell_info_dict[cell_id]
        faces = self.faces[cell_id]
        interfaces = []
        connections = []
        #in_plain_num = 0
        for face_id, face in faces.items():
            #t_face1 = np.average(self.mesh_data.points[cell.nodes_to_faces[face], :], axis=0) - cell.centroid
            #if not self.check_if_neumann_boundary(cell_id, face) or face == face_id:
            #prod = t_face1.dot(t_face) / np.linalg.norm(t_face) / np.linalg.norm(t_face1) + 1.0
            #if (abs(prod) > self.tol and not self.check_if_boundary(cell_id, face)) or face == face_id:
            #if (not self.check_if_boundary(cell_id, face)) or face == face_id:
            #if t_face1[2] == 0.0:
            #    if in_plain_num < 2:
            #        res.append(face)
            #        in_plain_num += 1
            #else:
            if face.type != FType.MAT_TO_FRAC:
                interfaces.append(face_id)
            connections.append(face_id)
        return np.array(connections, dtype=np.intp), np.array(interfaces, dtype=np.intp)
    def get_full_augmented_stencil(self, cell_id, face_id=0):
        cell = self.mat_cell_info_dict[cell_id]
        faces = self.faces[cell_id]
        interfaces = []
        connections = []
        #in_plain_num = 0
        for face_id, face in faces.items():
            if face.type != FType.MAT_TO_FRAC:
                interfaces.append(face_id)
            connections.append(face_id)

        adj_ids = []
        for node in cell.nodes_to_cell:
            for cell_id1 in self.mat_cells_to_node[node]:
                is_nebr = False
                for face in faces.values():
                    if cell_id1 == face.cell_id2 or cell_id1 == face.cell_id1:
                        is_nebr = True
                        break
                if not is_nebr and cell_id1 not in adj_ids:
                    adj_ids.append(cell_id1)

        return np.array(connections, dtype=np.intp), np.array(interfaces, dtype=np.intp), np.array(adj_ids, dtype=np.intp)
    def get_specific_stencil(self, cell_id, face_id):
        cell = self.mat_cell_info_dict[cell_id]
        faces = self.faces[cell_id]
        cur_face = faces[face_id]
        # Check if is_on_fault
        is_on_fault = (False, -1)
        for j, face in faces.items():
            if face.type == FType.MAT_TO_FRAC:
                is_on_fault[0] = True
                is_on_fault[1] = j
                break
        interfaces = [face_id]
        connections = [face_id]
        if is_on_fault[0]: connections.append(is_on_fault[1])
        in_plain_num = 1
        for id, face in faces.items():
            if id in connections: continue
            if np.fabs(cur_face.centroid[2] - face.centroid[2]) > 1.E-5:
                interfaces.append(id)
                connections.append(id)
            elif in_plain_num < 2:
                interfaces.append(id)
                connections.append(id)
                in_plain_num += 1
        return np.array(connections, dtype=np.intp), np.array(interfaces, dtype=np.intp)
    def get_frac_stencil(self, cell_id):
        faces = self.faces[cell_id]
        res = []
        for face_id, face in faces.items():
            if face.type == FType.FRAC:
                res.append(face.cell_id2)
        return res
    def calc_harmonic_flux(self, cells_id, faces_id):
        cells = [self.mat_cell_info_dict[cells_id[0]], self.mat_cell_info_dict[cells_id[1]]]
        face = self.faces[cells_id[0]][faces_id[0]]
        t_face1 = face.centroid - cells[0].centroid
        t_face2 = cells[1].centroid - face.centroid
        n = face.n
        if np.inner(t_face1, n) < 0: n = -n

        stf1 = self.stf[cells[0].prop_id]
        stf2 = self.stf[cells[1].prop_id]
        T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
        T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
        r1 = np.inner(t_face1, n)
        r2 = np.inner(t_face2, n)
        Tden = np.linalg.inv(r1 * T2 + r2 * T1)
        return Tden, T1.dot(Tden).dot(T2)
    def calc_disp_gradients(self, least_sq_sol, stencil):
        B = np.zeros((2 * 2 * self.n_dim, self.n_dim * self.n_dim, self.n_dim), dtype=np.float64)
        true_stencil = {}
        cell_pos = 0
        for id, data in stencil.items():
            if id not in true_stencil:
                true_stencil[id] = cell_pos
                B[cell_pos] = sum(least_sq_sol[:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in data)
                cell_pos += 1
            else:
                B[true_stencil[id]] += sum(least_sq_sol[:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in data)

        # sum_trans = np.abs(np.sum(B[:cell_pos], axis=0))
        # assert ((sum_trans < self.tol).all())
        non_zero_ind = (np.abs(B[:cell_pos]) > self.tol).any(axis=(1, 2))
        B[np.where(np.abs(B[:cell_pos]) < self.tol)] = 0.0
        return np.array(list(true_stencil.keys()))[non_zero_ind], B[:cell_pos][non_zero_ind]
    def get_harmonic_transversal_stiffness(self, cell_id, face_id):
        cell = self.mat_cell_info_dict[cell_id]
        face = self.faces[cell_id][face_id]
        t_face1 = face.centroid - cell.centroid
        n = face.n
        if np.inner(t_face1, n) < 0: n = -n
        stf1 = self.stf[cell.prop_id]

        cell_id2 = face.cell_id2
        if face.type == FType.MAT:
            cell2 = self.mat_cell_info_dict[cell_id2]
            t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
            stf2 = self.stf[cell2.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
            r1 = np.inner(t_face1, n)
            r2 = np.inner(t_face2, n)
            Tden = np.linalg.inv(r1 * T2 + r2 * T1)
            T = T1.dot(Tden).dot(T2)
            y1 = cell.centroid + r1 * n
            y2 = self.mat_cell_info_dict[cell_id2].centroid - r2 * n
            G = T.dot(np.kron(np.identity(self.n_dim), y1 - y2)) + \
                r1 * T2.dot(Tden).dot(G1) + r2 * T1.dot(Tden).dot(G2)
        if face.type == FType.BORDER:
            r1 = np.inner(t_face1, n)
            y1 = cell.centroid + r1 * n
            r2 = 0
            yb = face.centroid - r2 * n
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            T = T1 / r1
            G = T1.dot(np.kron(np.identity(self.n_dim), y1 - yb)) / r1 + G1
        return T, G
    def get_fracture_sign(self, t_face):
        face = self.faces[self.mat_cells_tot][4]
        prod = face.n.dot(t_face)
        return -1.0 if prod < 0.0 else 1.0
    def calc_transversal_fluxes_all_cells(self):
        self.transversal_fluxes = {}
        self.Gamma = {}
        n_dim = self.n_dim
        # Fracture gradient reconstruction
        for cell_id, cell in self.frac_cell_info_dict.items():
            frac_nebrs = self.get_frac_stencil(cell_id)
            stencil = {}
            R = np.zeros((len(frac_nebrs) * n_dim, n_dim * n_dim), dtype=np.float64)
            D = np.zeros((len(frac_nebrs) * n_dim, (len(frac_nebrs) + 1) * n_dim), dtype=np.float64)
            pos = 0
            for j, cell_id2 in enumerate(frac_nebrs):
                dx = self.frac_cell_info_dict[cell_id2].centroid - cell.centroid
                R[j * n_dim:(j + 1) * n_dim, :] = np.kron(np.identity(self.n_dim), dx)
                if cell_id not in stencil:
                    stencil[cell_id] = pos
                    pos += 1
                if cell_id2 not in stencil:
                    stencil[cell_id2] = pos
                    pos += 1

                D[j * self.n_dim:(j+1) * n_dim, stencil[cell_id2] * n_dim:(stencil[cell_id2]+1) * n_dim] = np.identity(n_dim)
                D[j * self.n_dim:(j+1) * n_dim, stencil[cell_id] * n_dim:(stencil[cell_id]+1) * n_dim] = -np.identity(n_dim)
                #if cell_id2 not in stencil:
                #    stencil[cell_id2] = []
                #stencil[cell_id2].append((j, np.identity(self.n_dim)))
                #if cell_id not in stencil:
                #    stencil[cell_id] = []
                #stencil[cell_id].append((j, -np.identity(self.n_dim)))

            sq_mat = R.T.dot(R)
            rank_sq = np.linalg.matrix_rank(sq_mat)
            if rank_sq < self.n_dim * self.n_dim:
                grads = np.linalg.pinv(R)
            else:
                grads = np.linalg.inv(sq_mat).dot(R.T)
            self.transversal_fluxes[cell_id] = (stencil, grads.dot(D))
        # Matrix gradient reconstruction
        for cell_id, cell in self.mat_cell_info_dict.items():
            # calculate harmonic & transversal stiffness
            self.Gamma[cell_id] = {}
            for face_id in cell.nodes_to_faces.keys():
                self.Gamma[cell_id][face_id] = self.get_harmonic_transversal_stiffness(cell_id, face_id)
            num, cur_faces = self.get_full_stencil(cell_id)
            stencil = {}
            R = np.zeros((num * self.n_dim, self.n_dim * self.n_dim), dtype=np.float64)
            for j, face_id in enumerate(cur_faces):
                face = self.faces[cell_id][face_id]
                t_face1 = face.centroid - cell.centroid
                n = face.n
                if np.inner(t_face1, n) < 0: n = -n
                stf1 = self.stf[cell.prop_id]

                cell_id2 = face.cell_id2
                if face.type == FType.MAT:
                    cell2 = self.mat_cell_info_dict[cell_id2]
                    t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]

                    T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r1 = np.inner(t_face1, n)
                    r2 = np.inner(t_face2, n)

                    tmp = np.array(
                        [np.concatenate(
                            (n.dot(stf1[0] - stf2[0]), n.dot((stf1[5] - stf2[5]).T), n.dot((stf1[4] - stf2[4]).T))),
                         np.concatenate(
                             (n.dot(stf1[5] - stf2[5]), n.dot(stf1[1] - stf2[1]), n.dot((stf1[3] - stf2[3]).T))),
                         np.concatenate(
                             (n.dot(stf1[4] - stf2[4]), n.dot(stf1[3] - stf2[3]), n.dot(stf1[2] - stf2[2])))])
                    dx = self.mat_cell_info_dict[cell_id2].centroid - cell.centroid
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), dx) + r2 * np.linalg.inv(T2).dot(tmp)
                    if cell_id2 not in stencil:
                        stencil[cell_id2] = []
                    stencil[cell_id2].append((j, np.identity(self.n_dim)))
                    if cell_id not in stencil:
                        stencil[cell_id] = []
                    stencil[cell_id].append((j, -np.identity(self.n_dim)))
                elif face.type == FType.BORDER:
                    # Boundary
                    bound_id = face.face_id2
                    bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
                    P = np.identity(self.n_dim)
                    tmp = np.array(
                        [np.concatenate((n.dot(stf1[0]), n.dot((stf1[5]).T), n.dot((stf1[4]).T))),
                         np.concatenate((n.dot(stf1[5]), n.dot(stf1[1]), n.dot((stf1[3]).T))),
                         np.concatenate((n.dot(stf1[4]), n.dot(stf1[3]), n.dot(stf1[2])))])
                    bound_num = self.mat_cells_tot + self.frac_cells_tot + bound_id
                    alpha = bc_data[0] * np.identity(self.n_dim)
                    beta = bc_data[1]
                    if bc_data[2] == self.Prol:
                        P -= np.outer(n, n)
                        alpha = np.outer(n, n)
                        beta = 1.0
                        if cell_id not in stencil:
                            stencil[cell_id] = []
                        stencil[cell_id].append((j, -alpha))
                    else:
                        if bound_num not in stencil:
                            stencil[bound_num] = []
                        stencil[bound_num].append((j, np.identity(self.n_dim)))
                        if (alpha != 0.0).any():
                            if cell_id not in stencil:
                                stencil[cell_id] = []
                            stencil[cell_id].append((j, -alpha))
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = alpha.dot(
                        np.kron(np.identity(self.n_dim), t_face1)) + beta * P.dot(tmp)
                elif face.type == FType.MAT_TO_FRAC:
                    j0 = np.argwhere(cur_faces == face.face_id1)[0][0]
                    face0 = self.faces[cell_id][cur_faces[j0]]

                    cell2 = self.mat_cell_info_dict[face0.cell_id2]
                    t_face2 = cell2.centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r2 = np.inner(t_face2, n)
                    y2 = cell2.centroid - r2 * n
                    data = self.transversal_fluxes[face.cell_id2]
                    tmp = -(np.kron(np.identity(self.n_dim), y2 - face.centroid) - r2 * np.linalg.inv(T2).dot(G2)).dot(data[1])

                    assert(face.cell_id1 == face0.cell_id1 and face.face_id1 == face0.face_id1)
                    sign = self.get_fracture_sign(-t_face1)
                    if face.cell_id2 not in stencil:
                        stencil[face.cell_id2] = []
                    stencil[face.cell_id2].append((j0, -sign * np.identity(n_dim)))

                    for id, pos in data[0].items():
                        if id not in stencil: stencil[id] = []
                        stencil[id].append((j0, sign * tmp[:, pos*n_dim:(pos+1)*n_dim]))

            sq_mat = R.T.dot(R)
            # rank_sq = np.linalg.matrix_rank(sq_mat)
            # rank = np.linalg.matrix_rank(R)
            # assert (rank_sq == sq_mat.shape[0])
            # cond = np.linalg.cond(sq_mat)
            grads = np.linalg.inv(sq_mat).dot(R.T)
            #self.disp_gradients[cell_id] = self.calc_disp_gradients(grads, stencil)
            self.transversal_fluxes[cell_id] = (stencil, grads)
    def calc_transversal_fluxes_all_cells_new(self):
        self.transversal_fluxes = {}
        n_dim = self.n_dim
        diag_id = np.arange(n_dim, dtype=np.intp)
        # Fracture gradient reconstruction
        for cell_id, cell in self.frac_cell_info_dict.items():
            self.transversal_fluxes[cell_id] = {}
            frac_nebrs = self.get_frac_stencil(cell_id)
            stencil = {}
            R = np.zeros((len(frac_nebrs) * n_dim, n_dim * n_dim), dtype=np.float64)
            D = np.zeros((len(frac_nebrs) * n_dim, (len(frac_nebrs) + 1) * n_dim), dtype=np.float64)
            pos = 0
            for j, cell_id2 in enumerate(frac_nebrs):
                dx = self.frac_cell_info_dict[cell_id2].centroid - cell.centroid
                R[j * n_dim:(j + 1) * n_dim, :] = np.kron(np.identity(self.n_dim), dx)
                if cell_id not in stencil:
                    stencil[cell_id] = pos
                    pos += 1
                if cell_id2 not in stencil:
                    stencil[cell_id2] = pos
                    pos += 1

                D[j * self.n_dim:(j+1) * n_dim, stencil[cell_id2] * n_dim:(stencil[cell_id2]+1) * n_dim] = np.identity(n_dim)
                D[j * self.n_dim:(j+1) * n_dim, stencil[cell_id] * n_dim:(stencil[cell_id]+1) * n_dim] = -np.identity(n_dim)

            sq_mat = R.T.dot(R)
            rank_sq = np.linalg.matrix_rank(sq_mat)
            if rank_sq < self.n_dim * self.n_dim:
                grads = np.linalg.pinv(R)
            else:
                grads = np.linalg.inv(sq_mat).dot(R.T)
            self.transversal_fluxes[cell_id][0] = (np.array(list(stencil.keys())), grads.dot(D))
        # Matrix gradient reconstruction
        for cell_id, cell in self.mat_cell_info_dict.items():
            self.transversal_fluxes[cell_id] = {}
            cur_conns, cur_faces = self.get_full_stencil(cell_id)
            num = cur_faces.size
            stencil = {}
            D = np.zeros((num * n_dim, (5 + num + 1) * n_dim), dtype=np.float64)
            R = np.zeros((num * n_dim, n_dim * n_dim), dtype=np.float64)
            W = np.zeros((num * n_dim, num * n_dim), dtype=np.float64)
            pos = 0
            # Choosing the stencil
            is_on_fault = 0
            for face in self.faces[cell_id].values():
                is_on_fault += (face.type == FType.MAT_TO_FRAC)
            for j, face_id in enumerate(cur_conns):
                face = self.faces[cell_id][face_id]
                t_face1 = face.centroid - cell.centroid
                n = face.n
                if np.inner(t_face1, n) < 0: n = -n
                stf1 = self.stf[cell.prop_id]

                cell_id2 = face.cell_id2
                if face.type == FType.MAT:
                    cell2 = self.mat_cell_info_dict[cell_id2]
                    t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]

                    T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r1 = np.inner(t_face1, n)
                    r2 = np.inner(t_face2, n)

                    tmp = np.array(
                        [np.concatenate(
                            (n.dot(stf1[0] - stf2[0]), n.dot((stf1[5] - stf2[5]).T), n.dot((stf1[4] - stf2[4]).T))),
                         np.concatenate(
                             (n.dot(stf1[5] - stf2[5]), n.dot(stf1[1] - stf2[1]), n.dot((stf1[3] - stf2[3]).T))),
                         np.concatenate(
                             (n.dot(stf1[4] - stf2[4]), n.dot(stf1[3] - stf2[3]), n.dot(stf1[2] - stf2[2])))])
                    dx = cell2.centroid - cell.centroid
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), dx) + r2 * np.linalg.inv(T2).dot(tmp)

                    if cell_id not in stencil:
                        stencil[cell_id] = pos
                        pos += 1
                    if cell_id2 not in stencil:
                        stencil[cell_id2] = pos
                        pos += 1

                    W[j * n_dim + diag_id,j * n_dim + diag_id] = 1.0 / np.linalg.norm(dx)
                    D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id2] * n_dim:(stencil[cell_id2] + 1) * n_dim] = np.identity(n_dim)
                    D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -np.identity(n_dim)
                elif face.type == FType.BORDER:
                    # Boundary
                    bound_id = face.face_id2
                    bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
                    P = np.identity(self.n_dim)
                    tmp = np.array(
                        [np.concatenate((n.dot(stf1[0]), n.dot((stf1[5]).T), n.dot((stf1[4]).T))),
                         np.concatenate((n.dot(stf1[5]), n.dot(stf1[1]), n.dot((stf1[3]).T))),
                         np.concatenate((n.dot(stf1[4]), n.dot(stf1[3]), n.dot(stf1[2])))])
                    bound_num = self.mat_cells_tot + self.frac_cells_tot + bound_id
                    alpha = bc_data[0] * np.identity(self.n_dim)
                    beta = bc_data[1]
                    if bc_data[2] == self.Prol:
                        P -= np.outer(n, n)
                        alpha = np.outer(n, n)
                        if cell_id not in stencil:
                            stencil[cell_id] = pos
                            pos += 1
                        D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                    else:
                        if bound_num not in stencil:
                            stencil[bound_num] = pos
                            pos += 1
                        D[j * self.n_dim:(j + 1) * n_dim,stencil[bound_num] * n_dim:(stencil[bound_num] + 1) * n_dim] = np.identity(n_dim)

                        if (alpha != 0.0).any():
                            if cell_id not in stencil:
                                stencil[cell_id] = pos
                                pos += 1
                            D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = alpha.dot(np.kron(np.identity(self.n_dim), t_face1)) + \
                                                                beta * P.dot(tmp)
                    W[j * n_dim + diag_id,j * n_dim + diag_id] = 1.0 / np.linalg.norm(face.centroid - cell.centroid)
                elif face.type == FType.MAT_TO_FRAC:
                    j0 = np.argwhere(cur_conns == face.face_id1)[0][0]
                    face0 = self.faces[cell_id][cur_conns[j0]]

                    cell2 = self.mat_cell_info_dict[face0.cell_id2]
                    t_face2 = cell2.centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r2 = np.inner(t_face2, n)
                    y2 = cell2.centroid - r2 * n
                    data = self.transversal_fluxes[face.cell_id2][0]
                    tmp = -(np.kron(np.identity(self.n_dim), y2 - face.centroid) - r2 * np.linalg.inv(T2).dot(G2)).dot(data[1])

                    assert(face.cell_id1 == face0.cell_id1 and face.face_id1 == face0.face_id1)
                    sign = self.get_fracture_sign(-t_face1)
                    # gap
                    if face.cell_id2 not in stencil:
                        stencil[face.cell_id2] = pos
                        pos += 1
                    D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[face.cell_id2] * n_dim:
                                                      (stencil[face.cell_id2] + 1) * n_dim] += -sign * np.identity(n_dim)
                    # gap gradients
                    for pos1, id in enumerate(data[0]):
                        if id not in stencil:
                            stencil[id] = pos
                            pos += 1
                        D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[id] * n_dim:
                                                (stencil[id] + 1) * n_dim] += sign * tmp[:, pos1*n_dim:(pos1+1)*n_dim]
            sq_mat = R.T.dot(W).dot(R)
            # rank_sq = np.linalg.matrix_rank(sq_mat)
            # rank = np.linalg.matrix_rank(R)
            # assert (rank_sq == sq_mat.shape[0])
            # cond = np.linalg.cond(sq_mat)
            grads = np.linalg.inv(sq_mat).dot(R.T).dot(W)
            self.transversal_fluxes[cell_id][0] = (np.array(list(stencil.keys())), grads.dot(D[:, :pos*n_dim]))
    def calc_transversal_fluxes_all_cells_new_homo_augmented(self):
        self.transversal_fluxes = {}
        n_dim = self.n_dim
        diag_id = np.arange(n_dim, dtype=np.intp)
        # Matrix gradient reconstruction
        for cell_id, cell in self.mat_cell_info_dict.items():
            self.transversal_fluxes[cell_id] = {}
            cur_conns, cur_faces, adj_cells = self.get_full_augmented_stencil(cell_id)
            num = cur_faces.size + adj_cells.size
            stencil = {}
            D = np.zeros((num * n_dim, (5 + num + 1) * n_dim), dtype=np.float64)
            R = np.zeros((num * n_dim, n_dim * n_dim), dtype=np.float64)
            W = np.zeros((num * n_dim, num * n_dim), dtype=np.float64)
            pos = 0
            # Choosing the stencil
            for j, face_id in enumerate(cur_conns):
                face = self.faces[cell_id][face_id]
                t_face1 = face.centroid - cell.centroid
                n = face.n
                if np.inner(t_face1, n) < 0: n = -n
                stf1 = self.stf[cell.prop_id]

                cell_id2 = face.cell_id2
                if face.type == FType.MAT:
                    cell2 = self.mat_cell_info_dict[cell_id2]
                    dx = cell2.centroid - cell.centroid
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), dx)

                    if cell_id not in stencil:
                        stencil[cell_id] = pos
                        pos += 1
                    if cell_id2 not in stencil:
                        stencil[cell_id2] = pos
                        pos += 1

                    W[j * n_dim + diag_id,j * n_dim + diag_id] = 1.0# / np.linalg.norm(dx)
                    D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id2] * n_dim:(stencil[cell_id2] + 1) * n_dim] = np.identity(n_dim)
                    D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -np.identity(n_dim)
                elif face.type == FType.BORDER:
                    # Boundary
                    bound_id = face.face_id2
                    bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
                    P = np.identity(self.n_dim)
                    tmp = np.array(
                        [np.concatenate((n.dot(stf1[0]), n.dot((stf1[5]).T), n.dot((stf1[4]).T))),
                         np.concatenate((n.dot(stf1[5]), n.dot(stf1[1]), n.dot((stf1[3]).T))),
                         np.concatenate((n.dot(stf1[4]), n.dot(stf1[3]), n.dot(stf1[2])))])
                    bound_num = self.mat_cells_tot + self.frac_cells_tot + bound_id
                    alpha = bc_data[0] * np.identity(self.n_dim)
                    beta = bc_data[1]
                    if bc_data[2] == self.Prol:
                        P -= np.outer(n, n)
                        alpha = np.outer(n, n)
                        if cell_id not in stencil:
                            stencil[cell_id] = pos
                            pos += 1
                        D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                    else:
                        if bound_num not in stencil:
                            stencil[bound_num] = pos
                            pos += 1
                        D[j * self.n_dim:(j + 1) * n_dim,stencil[bound_num] * n_dim:(stencil[bound_num] + 1) * n_dim] = np.identity(n_dim)

                        if (alpha != 0.0).any():
                            if cell_id not in stencil:
                                stencil[cell_id] = pos
                                pos += 1
                            D[j * self.n_dim:(j + 1) * n_dim,stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                    R[j * self.n_dim:(j + 1) * self.n_dim, :] = alpha.dot(np.kron(np.identity(self.n_dim), t_face1)) + \
                                                                beta * P.dot(tmp)
                    W[j * n_dim + diag_id,j * n_dim + diag_id] = 1.0# / np.linalg.norm(face.centroid - cell.centroid)
                elif face.type == FType.MAT_TO_FRAC:
                    j0 = np.argwhere(cur_conns == face.face_id1)[0][0]
                    face0 = self.faces[cell_id][cur_conns[j0]]

                    cell2 = self.mat_cell_info_dict[face0.cell_id2]
                    t_face2 = cell2.centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r2 = np.inner(t_face2, n)
                    y2 = cell2.centroid - r2 * n
                    data = self.transversal_fluxes[face.cell_id2][0]
                    tmp = -(np.kron(np.identity(self.n_dim), y2 - face.centroid) - r2 * np.linalg.inv(T2).dot(G2)).dot(data[1])

                    assert(face.cell_id1 == face0.cell_id1 and face.face_id1 == face0.face_id1)
                    sign = self.get_fracture_sign(-t_face1)
                    # gap
                    if face.cell_id2 not in stencil:
                        stencil[face.cell_id2] = pos
                        pos += 1
                    D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[face.cell_id2] * n_dim:
                                                      (stencil[face.cell_id2] + 1) * n_dim] += -sign * np.identity(n_dim)
                    # gap gradients
                    for pos1, id in enumerate(data[0]):
                        if id not in stencil:
                            stencil[id] = pos
                            pos += 1
                        D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[id] * n_dim:
                                                (stencil[id] + 1) * n_dim] += sign * tmp[:, pos1*n_dim:(pos1+1)*n_dim]
            # Augmented
            for j0, cell_id2 in enumerate(adj_cells):
                j = j0 + cur_faces.size
                cell2 = self.mat_cell_info_dict[cell_id2]
                dx = cell2.centroid - cell.centroid

                if cell_id not in stencil:
                    stencil[cell_id] = pos
                    pos += 1
                if cell_id2 not in stencil:
                    stencil[cell_id2] = pos
                    pos += 1

                R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), dx)
                W[j * n_dim + diag_id, j * n_dim + diag_id] = 1.0# / np.linalg.norm(dx)
                D[j * self.n_dim:(j + 1) * n_dim, stencil[cell_id2] * n_dim:(stencil[cell_id2] + 1) * n_dim] = np.identity(
                    n_dim)
                D[j * self.n_dim:(j + 1) * n_dim, stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -np.identity(
                    n_dim)
            sq_mat = R.T.dot(W).dot(R)
            # rank_sq = np.linalg.matrix_rank(sq_mat)
            # rank = np.linalg.matrix_rank(R)
            # assert (rank_sq == sq_mat.shape[0])
            # cond = np.linalg.cond(sq_mat)
            grads = np.linalg.inv(sq_mat).dot(R.T).dot(W)
            self.transversal_fluxes[cell_id][0] = (np.array(list(stencil.keys())), grads.dot(D[:, :pos*n_dim]))
    def calc_transversal_fluxes_all_cells_all_faces(self):
        self.transversal_fluxes = {}
        n_dim = self.n_dim
        # Fracture gradient reconstruction
        for cell_id, cell in self.frac_cell_info_dict.items():
            self.transversal_fluxes[cell_id] = {}
            frac_nebrs = self.get_frac_stencil(cell_id)
            stencil = {}
            R = np.zeros((len(frac_nebrs) * n_dim, n_dim * n_dim), dtype=np.float64)
            D = np.zeros((len(frac_nebrs) * n_dim, (len(frac_nebrs) + 1) * n_dim), dtype=np.float64)
            pos = 0
            for j, cell_id2 in enumerate(frac_nebrs):
                dx = self.frac_cell_info_dict[cell_id2].centroid - cell.centroid
                R[j * n_dim:(j + 1) * n_dim, :] = np.kron(np.identity(self.n_dim), dx)
                if cell_id not in stencil:
                    stencil[cell_id] = pos
                    pos += 1
                if cell_id2 not in stencil:
                    stencil[cell_id2] = pos
                    pos += 1

                D[j * self.n_dim:(j + 1) * n_dim,
                stencil[cell_id2] * n_dim:(stencil[cell_id2] + 1) * n_dim] = np.identity(n_dim)
                D[j * self.n_dim:(j + 1) * n_dim,
                stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -np.identity(n_dim)

            sq_mat = R.T.dot(R)
            rank_sq = np.linalg.matrix_rank(sq_mat)
            if rank_sq < self.n_dim * self.n_dim:
                grads = np.linalg.pinv(R)
            else:
                grads = np.linalg.inv(sq_mat).dot(R.T)
            self.transversal_fluxes[cell_id][0] = (np.array(list(stencil.keys())), grads.dot(D))
        # Matrix gradient reconstruction
        for cell_id, cell in self.mat_cell_info_dict.items():
            self.transversal_fluxes[cell_id] = {}
            all_conns, all_faces = self.get_full_stencil(cell_id)
            # Specific reconstruction for each face
            for face_id0 in all_faces:
                cur_conns, cur_faces = self.get_not_coplanar_stencil(cell_id, face_id0)
                num = cur_faces.size
                stencil = {}
                D = np.zeros((num * n_dim, (5 + num + 1) * n_dim), dtype=np.float64)
                R = np.zeros((num * n_dim, n_dim * n_dim), dtype=np.float64)
                pos = 0
                # Choosing the stencil
                is_on_fault = 0
                for face in self.faces[cell_id].values():
                    is_on_fault += (face.type == FType.MAT_TO_FRAC)
                for j, face_id in enumerate(cur_conns):
                    face = self.faces[cell_id][face_id]
                    t_face1 = face.centroid - cell.centroid
                    n = face.n
                    if np.inner(t_face1, n) < 0: n = -n
                    stf1 = self.stf[cell.prop_id]

                    cell_id2 = face.cell_id2
                    if face.type == FType.MAT:
                        cell2 = self.mat_cell_info_dict[cell_id2]
                        t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
                        stf2 = self.stf[cell2.prop_id]

                        T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                        T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                        r1 = np.inner(t_face1, n)
                        r2 = np.inner(t_face2, n)

                        tmp = np.array(
                            [np.concatenate(
                                (n.dot(stf1[0] - stf2[0]), n.dot((stf1[5] - stf2[5]).T), n.dot((stf1[4] - stf2[4]).T))),
                                np.concatenate(
                                    (n.dot(stf1[5] - stf2[5]), n.dot(stf1[1] - stf2[1]), n.dot((stf1[3] - stf2[3]).T))),
                                np.concatenate(
                                    (n.dot(stf1[4] - stf2[4]), n.dot(stf1[3] - stf2[3]), n.dot(stf1[2] - stf2[2])))])
                        dx = self.mat_cell_info_dict[cell_id2].centroid - cell.centroid
                        R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), dx) + r2 * np.linalg.inv(T2).dot(tmp)

                        if cell_id not in stencil:
                            stencil[cell_id] = pos
                            pos += 1
                        if cell_id2 not in stencil:
                            stencil[cell_id2] = pos
                            pos += 1

                        D[j * self.n_dim:(j + 1) * n_dim,
                        stencil[cell_id2] * n_dim:(stencil[cell_id2] + 1) * n_dim] = np.identity(n_dim)
                        D[j * self.n_dim:(j + 1) * n_dim,
                        stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -np.identity(n_dim)
                    elif face.type == FType.BORDER:
                        # Boundary
                        bound_id = face.face_id2
                        bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
                        P = np.identity(self.n_dim)
                        tmp = np.array(
                            [np.concatenate((n.dot(stf1[0]), n.dot((stf1[5]).T), n.dot((stf1[4]).T))),
                             np.concatenate((n.dot(stf1[5]), n.dot(stf1[1]), n.dot((stf1[3]).T))),
                             np.concatenate((n.dot(stf1[4]), n.dot(stf1[3]), n.dot(stf1[2])))])
                        bound_num = self.mat_cells_tot + self.frac_cells_tot + bound_id
                        alpha = bc_data[0] * np.identity(self.n_dim)
                        beta = bc_data[1]
                        if bc_data[2] == self.Prol:
                            P -= np.outer(n, n)
                            alpha = np.outer(n, n)
                            if cell_id not in stencil:
                                stencil[cell_id] = pos
                                pos += 1
                            D[j * self.n_dim:(j + 1) * n_dim,
                            stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                        else:
                            if bound_num not in stencil:
                                stencil[bound_num] = pos
                                pos += 1
                            D[j * self.n_dim:(j + 1) * n_dim,
                            stencil[bound_num] * n_dim:(stencil[bound_num] + 1) * n_dim] = np.identity(n_dim)

                            if (alpha != 0.0).any():
                                if cell_id not in stencil:
                                    stencil[cell_id] = pos
                                    pos += 1
                                D[j * self.n_dim:(j + 1) * n_dim,
                                stencil[cell_id] * n_dim:(stencil[cell_id] + 1) * n_dim] = -alpha
                        R[j * self.n_dim:(j + 1) * self.n_dim, :] = alpha.dot(
                            np.kron(np.identity(self.n_dim), t_face1)) + beta * P.dot(tmp)
                    elif face.type == FType.MAT_TO_FRAC:
                        j0 = np.argwhere(cur_conns == face.face_id1)[0][0]
                        face0 = self.faces[cell_id][cur_conns[j0]]

                        cell2 = self.mat_cell_info_dict[face0.cell_id2]
                        t_face2 = cell2.centroid - face.centroid
                        stf2 = self.stf[cell2.prop_id]
                        T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                        r2 = np.inner(t_face2, n)
                        y2 = cell2.centroid - r2 * n
                        data = self.transversal_fluxes[face.cell_id2][0]
                        tmp = -(np.kron(np.identity(self.n_dim), y2 - face.centroid) - r2 * np.linalg.inv(
                            T2).dot(G2)).dot(data[1])

                        assert (face.cell_id1 == face0.cell_id1 and face.face_id1 == face0.face_id1)
                        sign = self.get_fracture_sign(-t_face1)
                        # gap
                        if face.cell_id2 not in stencil:
                            stencil[face.cell_id2] = pos
                            pos += 1
                        D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[face.cell_id2] * n_dim:
                                                            (stencil[face.cell_id2] + 1) * n_dim] += -sign * np.identity(
                            n_dim)
                        # gap gradients
                        for pos1, id in enumerate(data[0]):
                            if id not in stencil:
                                stencil[id] = pos
                                pos += 1
                            D[j0 * self.n_dim:(j0 + 1) * n_dim, stencil[id] * n_dim:
                                                                (stencil[id] + 1) * n_dim] += sign * tmp[:, pos1 * n_dim:(
                                                                                                                                     pos1 + 1) * n_dim]
                sq_mat = R.T.dot(R)
                # rank_sq = np.linalg.matrix_rank(sq_mat)
                # rank = np.linalg.matrix_rank(R)
                # assert (rank_sq == sq_mat.shape[0])
                # cond = np.linalg.cond(sq_mat)
                grads = np.linalg.inv(sq_mat).dot(R.T)
                #self.transversal_fluxes[cell_id][face_id0] = (np.array(list(stencil.keys())), grads.dot(D[:, :pos * n_dim]))

                face = self.faces[cell_id][face_id0]
                t_face1 = face.centroid - cell.centroid
                n = face.n
                if np.inner(t_face1, n) < 0: n = -n
                stf1 = self.stf[cell.prop_id]

                cell_id2 = face.cell_id2
                if face.type != FType.BORDER:
                    Tden, T = self.calc_harmonic_flux([cell_id, cell_id2], [face_id0, face.face_id2])

                if cell_id != cell_id2:
                    cell2 = self.mat_cell_info_dict[cell_id2]
                    t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
                    stf2 = self.stf[cell2.prop_id]
                    T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                    T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                    r1 = np.inner(t_face1, n)
                    r2 = np.inner(t_face2, n)
                    y1 = cell.centroid + r1 * n
                    y2 = self.mat_cell_info_dict[cell_id2].centroid - r2 * n
                    G = T.dot(np.kron(np.identity(self.n_dim), y1-y2)) + \
                        r1 * T2.dot(Tden).dot(G1) + r2 * T1.dot(Tden).dot(G2)
                else:
                    r1 = np.inner(t_face1, n)
                    y1 = cell.centroid + r1 * n
                    r2 = 0
                    yb = face.centroid - r2 * n
                    T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                    G = T1.dot(np.kron(np.identity(self.n_dim), y1 - yb)) / r1 + G1

                G = G.dot(np.kron(np.identity(self.n_dim), np.identity(self.n_dim) - np.outer(n, n)))
                self.transversal_fluxes[cell_id][face_id0] = np.array(list(stencil.keys()), dtype=np.intp), G.dot(grads.dot(D[:, :pos * n_dim]))
    def get_transversal_flux(self, cell_id, face_id, T, Tden):
        cell = self.mat_cell_info_dict[cell_id]
        face = self.faces[cell_id][face_id]
        t_face1 = face.centroid - cell.centroid
        n = face.n
        if np.inner(t_face1, n) < 0: n = -n
        stf1 = self.stf[cell.prop_id]

        cell_id2 = face.cell_id2
        if cell_id != cell_id2:
            cell2 = self.mat_cell_info_dict[cell_id2]
            t_face2 = self.mat_cell_info_dict[cell_id2].centroid - face.centroid
            stf2 = self.stf[cell2.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
            r1 = np.inner(t_face1, n)
            r2 = np.inner(t_face2, n)
            y1 = cell.centroid + r1 * n
            y2 = self.mat_cell_info_dict[cell_id2].centroid - r2 * n
            G = T.dot(np.kron(np.identity(self.n_dim), y1 - y2)) + \
                r1 * T2.dot(Tden).dot(G1) + r2 * T1.dot(Tden).dot(G2)
        else:
            r1 = np.inner(t_face1, n)
            y1 = cell.centroid + r1 * n
            r2 = 0
            yb = face.centroid - r2 * n
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            G = T1.dot(np.kron(np.identity(self.n_dim), y1 - yb)) / r1 + G1
        G = G.dot(np.kron(np.identity(self.n_dim), np.identity(self.n_dim) - np.outer(n, n)))
        data = self.transversal_fluxes[cell_id][0]
        return data[0], G.dot(data[1])
    def calc_transversal_flux(self, cell_id, face_id, T, Tden):
        G = np.zeros((self.n_dim, self.n_dim * self.n_dim), dtype=np.float64)
        stencil = {}
        cell = self.mat_cell_info_dict[cell_id]
        stf1 = self.stf[cell.prop_id]
        #cur_faces = self.get_not_coplanar_stencil(cell_id, face_id)
        cur_faces = self.get_full_stencil(cell_id, face_id)
        R = np.zeros((len(cur_faces) * self.n_dim, self.n_dim * self.n_dim), dtype=np.float64)
        for j, face_id1 in enumerate(cur_faces):
            face = self.faces[cell_id][face_id1]
            t_face1 = face.centroid - cell.centroid
            n = face.n
            if np.inner(t_face1, n) < 0: n = -n

            cell_id2 = face.cell_id2
            if cell_id != cell_id2:
                cell2 = self.mat_cell_info_dict[cell_id2]
                t_face2 = cell2.centroid - face.centroid
                stf2 = self.stf[cell2.prop_id]

                T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
                r1 = np.inner(t_face1, n)
                r2 = np.inner(t_face2, n)
                if face_id1 == face_id:
                    y1 = cell.centroid + r1 * n
                    y2 = self.mat_cell_info_dict[cell_id2].centroid - r2 * n
                    G = T.dot(np.kron(np.identity(self.n_dim), y1 - y2)) + \
                           r1 * T2.dot(Tden).dot(G1) + r2 * T1.dot(Tden).dot(G2)
                    G = G.dot(np.kron(np.identity(self.n_dim), np.identity(self.n_dim) - np.outer(n, n)))
                tmp = np.array(
                    [np.concatenate((n.dot(stf1[0] - stf2[0]), n.dot((stf1[5] - stf2[5]).T), n.dot((stf1[4] - stf2[4]).T))),
                     np.concatenate((n.dot(stf1[5] - stf2[5]), n.dot(stf1[1] - stf2[1]), n.dot((stf1[3] - stf2[3]).T))),
                     np.concatenate((n.dot(stf1[4] - stf2[4]), n.dot(stf1[3] - stf2[3]), n.dot(stf1[2] - stf2[2])))])
                R[j * self.n_dim:(j + 1) * self.n_dim, :] = np.kron(np.identity(self.n_dim), self.mat_cell_info_dict[cell_id2].centroid -
                                                                        cell.centroid) + r2 * np.linalg.inv(T2).dot(tmp)
                if cell_id2 not in stencil:
                    stencil[cell_id2] = []
                stencil[cell_id2].append((j, 1.0))
                if cell_id not in stencil:
                    stencil[cell_id] = []
                stencil[cell_id].append((j, -1.0))
            else:
                # Boundary
                bound_id = face.face_id2
                bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
                P = np.identity(self.n_dim)
                if bc_data[2] == self.Prol:
                    P -= np.outer(n, n)
                tmp = np.array(
                    [np.concatenate((n.dot(stf1[0]), n.dot((stf1[5]).T), n.dot((stf1[4]).T))),
                     np.concatenate((n.dot(stf1[5]), n.dot(stf1[1]), n.dot((stf1[3]).T))),
                     np.concatenate((n.dot(stf1[4]), n.dot(stf1[3]), n.dot(stf1[2])))])
                R[j * self.n_dim:(j + 1) * self.n_dim, :] = bc_data[0] * np.kron(np.identity(self.n_dim), t_face1) + bc_data[1] * P.dot(tmp)
                bound_num = self.mat_cells_tot + bound_id
                if bound_num not in stencil:
                    stencil[bound_num] = []
                stencil[bound_num].append((j, 1.0))
                if bc_data[0] != 0.0:
                    if cell_id not in stencil:
                        stencil[cell_id] = []
                    stencil[cell_id].append((j, -bc_data[0]))

                if face_id1 == face_id:
                    r1 = np.inner(t_face1, n)
                    y1 = cell.centroid + r1 * n
                    r2 = 0
                    yb = face.centroid - r2 * n
                    T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
                    G = T1.dot(np.kron(np.identity(self.n_dim),y1 - yb)) / r1 + G1
                    G = G.dot(np.kron(np.identity(self.n_dim), np.identity(self.n_dim) - np.outer(n, n)))

        sq_mat = R.T.dot(R)
        #rank_sq = np.linalg.matrix_rank(sq_mat)
        #rank = np.linalg.matrix_rank(R)
        #assert (rank_sq == sq_mat.shape[0])
        #cond = np.linalg.cond(sq_mat)
        grads = np.linalg.inv(sq_mat).dot(R.T)
        # Reconstruction of displacement gradients
        cell_id1 = self.faces[cell_id][face_id].cell_id2
        if cell_id not in self.disp_gradients and cell_id1 != cell_id:
            self.disp_gradients[cell_id] = self.calc_disp_gradients(grads, stencil)

        return stencil, G.dot(grads)
    def write_mpsa_conn_to_file(self, path='mpsa_conn.dat'):
        f = open(path, 'w')
        f.write(str(self.mpsa_connections_num) + '\n')
        for cell_id, faces in self.mpsa_connections.items():
            for face_id, data in faces.items():
                cell_id1 = data[0][0]
                isBound = cell_id < self.mat_cells_tot and cell_id == cell_id1
                isFrac = cell_id >= self.mat_cells_tot
                if isBound:
                    f.write(str(cell_id) + '\t' + str(self.mat_cells_tot + self.frac_cells_tot + self.faces[cell_id][face_id].face_id2) + '\n')
                else:
                    f.write(str(cell_id) + '\t' + str(cell_id1) + '\n')

                ids = np.argsort(data[1])
                for k in range(self.n_dim):
                    row = 'F' + str(k)
                    for i, id in enumerate(data[1][ids]):  row += '\t' + str(id) + '\t[' + ', '.join(['{:.2e}'.format(n) for n in data[2][ids][i,k,:]]) + str(']')
                    f.write(row + '\n')
        f.close()
    def merge_fluxes(self, st_from, Ffrom, st_to, Fto):
        if st_from.size > 0:
            for i, cell_id in enumerate(st_from):
                idx = np.where(st_to == cell_id)
                if idx[0].size > 0:
                    Fto[idx[0][0]] += Ffrom[i]
                else:
                    st_to = np.append(st_to, cell_id)
                    Fto = np.vstack([Fto, Ffrom[i][np.newaxis]])
        return st_to, Fto
    def merge_connection(self, cell_id, face_id):
        nebr = (cell_id, face_id)
        stencil = np.array([], dtype=np.int32)
        F = np.empty(shape=(0,self.n_dim,self.n_dim))
        face = self.faces[cell_id][face_id]
        mult = 1.0 if face.type == FType.BORDER else 0.5

        if face_id in self.Fharm[cell_id]:
            nebr, st_harm, Fharm = self.Fharm[cell_id][face_id]
            stencil, F = self.merge_fluxes(st_harm, Fharm, stencil, F)
        if face_id in self.Ft1[cell_id]:
            nebr, st_t1, Ft1 = self.Ft1[cell_id][face_id]
            stencil, F = self.merge_fluxes(st_t1, mult * Ft1, stencil, F)
        if face_id in self.Ft2[cell_id]:
            nebr, st_t2, Ft2 = self.Ft2[cell_id][face_id]
            stencil, F = self.merge_fluxes(st_t2, mult * Ft2, stencil, F)
        if cell_id in self.Fcont1:
            if face_id in self.Fcont1[cell_id]:
                nebr, st_c1, Fc1 = self.Fcont1[cell_id][face_id]
                stencil, F = self.merge_fluxes(st_c1, Fc1 / 2, stencil, F)
        if cell_id in self.Fcont2:
            if face_id in self.Fcont2[cell_id]:
                nebr, st_c2, Fc2 = self.Fcont2[cell_id][face_id]
                stencil, F = self.merge_fluxes(st_c2, Fc2 / 2, stencil, F)

        return nebr, stencil, F
    def calc_mpsa_connection(self, cell_id1, face_id1, cell_id2, face_id2):
        face = self.faces[cell_id1][face_id1]
        n_dim = self.n_dim
        cell_pos = 0
        stencil = [{}, {}]
        stenc_cells = {}
        A = []
        B = np.zeros(((2 * 2 + 5) * self.n_dim, self.n_dim, self.n_dim), dtype=np.float64)
        if face.type == FType.MAT:
            # Calculate harmonic parts of the flux for both cells
            cells_id = [cell_id1, cell_id2]
            faces_id = [face_id1, face_id2]
            Tden, T = self.calc_harmonic_flux(cells_id, faces_id)
            # Calculate transversal parts of the flux for both cells
            for i in range(2):
                Tbuf = T if i == 0 else T.T
                #stencil[i], a = self.calc_transversal_flux(cells_id[i], faces_id[i], Tbuf, Tden)
                stencil[i], a = self.get_transversal_flux(cells_id[i], faces_id[i], Tbuf, Tden)
                A.append(((-1) ** i) * a)
            # Assemble stencil for transversal flux
            # List of final stencil for stress flux approximation over interface
            for i in range(2):
                for id, coef in stencil[i].items():
                    if id not in stenc_cells:
                        stenc_cells[id] = cell_pos
                        B[cell_pos] = 0.5 * sum(A[i][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)
                        cell_pos += 1
                    else:
                        B[stenc_cells[id]] += 0.5 * sum(A[i][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)
            if cells_id[1] not in stenc_cells:
                stenc_cells[cells_id[1]] = cell_pos
                cell_pos += 1
            B[stenc_cells[cells_id[1]]] += T
            B[stenc_cells[cells_id[0]]] -= T
        elif face.type == FType.BORDER:
            cell = self.mat_cell_info_dict[cell_id1]
            t_face1 = face.centroid - cell.centroid
            n = face.n
            if np.inner(t_face1, n) < 0: n = -n
            stf1 = self.stf[cell.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            r1 = np.inner(t_face1, n)

            bound_id = self.faces[cell_id1][face_id1].face_id2
            bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
            assert (abs(bc_data[0]) + abs(bc_data[1]) > 0)
            P = np.identity(self.n_dim)

            #stencil[0], a = self.calc_transversal_flux(cell_id1, face_id1, 0.0, 0.0)
            stencil[0], a = self.get_transversal_flux(cell_id1, face_id1, 0.0, 0.0)
            A.append(a)

            if bc_data[2] == self.P12:  # Dirichlet / Neumann boundary
                tmp = np.linalg.inv(bc_data[0] * np.identity(self.n_dim) + bc_data[1] * P.dot(T1) / r1)
                T = T1.dot(tmp) / r1
                if bc_data[0] != 0.0:
                    A[0] = (np.identity(self.n_dim) - bc_data[1] / r1 * (T1.dot(tmp)).dot(P)).dot(A[0])
                    # Assemble stencil for transversal flux
                    for id, coef in stencil[0].items():
                        if id not in stenc_cells:
                            stenc_cells[id] = cell_pos
                            B[cell_pos] = sum(A[0][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)
                            cell_pos += 1
                        else:
                            B[stenc_cells[id]] += sum(A[0][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)

                    B[stenc_cells[cell_id1]] -= bc_data[0] * T
                if bound_id + self.mat_cells_tot + self.frac_cells_tot not in stenc_cells:
                    stenc_cells[bound_id + self.mat_cells_tot + self.frac_cells_tot] = cell_pos
                    cell_pos += 1
                B[stenc_cells[bound_id + self.mat_cells_tot + self.frac_cells_tot]] += T
            else:  # Roller boundary
                T1inv = np.linalg.inv(T1)
                tmp = np.outer(n, n) / n.dot(T1inv).dot(n)
                A[0] = tmp.dot(T1inv).dot(A[0])
                # Assemble stencil for transversal flux
                for id, coef in stencil[0].items():
                    if id not in stenc_cells:
                        stenc_cells[id] = cell_pos
                        B[cell_pos] = sum(A[0][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)
                        cell_pos += 1
                    else:
                        B[stenc_cells[id]] += sum(A[0][:, j * self.n_dim:(j + 1) * self.n_dim].dot(mult) for j, mult in coef)

                B[stenc_cells[cell_id1]] -= tmp / r1
        elif face.type == FType.MAT_TO_FRAC:
            face0 = self.faces[cell_id1][face.face_id1]

            cell = self.mat_cell_info_dict[cell_id1]
            t_face1 = face.centroid - cell.centroid
            n = face.n
            if np.inner(t_face1, n) < 0: n = -n
            stf1 = self.stf[cell.prop_id]

            cell2 = self.mat_cell_info_dict[face0.cell_id2]
            t_face2 = cell2.centroid - face.centroid
            stf2 = self.stf[cell2.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
            r1 = np.inner(t_face1, n)
            r2 = np.inner(t_face2, n)
            Tden = np.linalg.inv(r1 * T2 + r2 * T1)
            T = T1.dot(Tden).dot(T2)
            y1 = cell.centroid + r1 * n
            y2 = cell2.centroid - r2 * n
            G = T.dot(np.kron(np.identity(self.n_dim), face.centroid - y2)) + \
                T.T.dot(np.kron(np.identity(self.n_dim), y1 - face.centroid)) + \
                r2 * T1.dot(Tden).dot(G2) - r1 * T2.dot(Tden).dot(G1)

            t_face1 = face.centroid - self.mat_cell_info_dict[cell_id1].centroid
            sign = self.get_fracture_sign(-t_face1)
            conn0, conn1, conn2 = self.mpsa_connections[cell_id1][face.face_id1]
            pos = 0
            if cell_id2 not in conn1:
                conn1 = np.append(conn1, cell_id2)
                conn2 = np.vstack([conn2, -sign * face.area * T[np.newaxis,:,:]])
            else:
                pos = np.argwhere(conn1 == cell_id2)[0][0]
                conn2[pos] += -sign * face.area * T

            data = self.transversal_fluxes[cell_id2]
            #tmp = self.Gamma[cell_id1][face.face_id1][1].dot(data[1]) / 2.0
            tmp = G.dot(data[1]) / 2.0
            for id, pos in data[0].items():
                if id not in conn1:
                    conn1 = np.append(conn1, id)
                    conn2 = np.append(conn2, (sign * face.area * tmp[:, pos * n_dim:(pos + 1) * n_dim])[np.newaxis,:], axis=0)
                else:
                    conn2[conn1 == id][0] += sign * face.area * tmp[:, pos * n_dim:(pos + 1) * n_dim]

            self.mpsa_connections[cell_id1][face.face_id1] = (conn0, conn1, conn2)
        # Check if the sum of transmissibilities is equal to zero
        sum_trans = np.abs(np.sum(B[:cell_pos], axis=0))
        diff_identity = np.abs(sum_trans - np.identity(self.n_dim))
        if not (sum_trans < self.tol).all() and not (cell_id1 == cell_id2 and (diff_identity < self.tol).all()):
            aaa = 555
        #assert ((sum_trans < self.tol).all() or (isBound and (diff_identity < self.tol).all()))
        non_zero_ind = (np.abs(B[:cell_pos]) > self.tol).any(axis=(1, 2))
        B[np.where(np.abs(B[:cell_pos]) < self.tol)] = 0.0
        return (cell_id2, face_id2), np.array(list(stenc_cells.keys()))[non_zero_ind], face.area * B[:cell_pos][non_zero_ind]
    def calc_mpsa_connection_by_terms(self, cell_id1, face_id1, cell_id2, face_id2):
        face = self.faces[cell_id1][face_id1]
        n_dim = self.n_dim

        if cell_id1 not in self.Fharm:
            self.Fharm[cell_id1] = {}
        if cell_id1 not in self.Ft1:
            self.Ft1[cell_id1] = {}
        if cell_id1 not in self.Ft2:
            self.Ft2[cell_id1] = {}

        if face.type == FType.MAT:
            # Calculate harmonic parts of the flux for both cells
            cells_id = [cell_id1, cell_id2]
            faces_id = [face_id1, face_id2]
            Tden, T = self.calc_harmonic_flux(cells_id, faces_id)
            self.Fharm[cell_id1][face_id1] = ((cell_id2, face_id2), np.array([cell_id1, cell_id2]),
                                              face.area * np.array([-T, T]))
            # Calculate transversal parts of the flux for both cells
            stencil, a = self.get_transversal_flux(cell_id1, face_id1, T, Tden)
            #stencil, a = self.transversal_fluxes[cell_id1][face_id1]
            a = np.array(np.hsplit(a, stencil.size))
            non_zero_ind = (np.fabs(a) > self.tol).any(axis=(1, 2))
            self.Ft1[cell_id1][face_id1] = ((cell_id2, face_id2), stencil[non_zero_ind], face.area * a[non_zero_ind])

            stencil, a = self.get_transversal_flux(cell_id2, face_id2, T.T, Tden)
            #stencil, a = self.transversal_fluxes[cell_id2][face_id2]
            a = np.array(np.hsplit(a,stencil.size))
            non_zero_ind = (np.fabs(a) > self.tol).any(axis=(1, 2))
            self.Ft2[cell_id1][face_id1] = ((cell_id2, face_id2), stencil[non_zero_ind], -face.area * a[non_zero_ind])

            self.Fharm[cell_id1][face_id1][2][np.where(np.fabs(self.Fharm[cell_id1][face_id1][2]) < self.tol)] = 0.0
            self.Ft1[cell_id1][face_id1][2][np.where(np.fabs(self.Ft1[cell_id1][face_id1][2]) < self.tol)] = 0.0
            self.Ft2[cell_id1][face_id1][2][np.where(np.fabs(self.Ft2[cell_id1][face_id1][2]) < self.tol)] = 0.0
        elif face.type == FType.BORDER:
            cell = self.mat_cell_info_dict[cell_id1]
            t_face1 = face.centroid - cell.centroid
            n = face.n
            if np.inner(t_face1, n) < 0: n = -n
            stf1 = self.stf[cell.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            r1 = np.inner(t_face1, n)

            bound_id = self.faces[cell_id1][face_id1].face_id2
            bc_data = self.bc_mech[self.bcm_num * bound_id:self.bcm_num * (bound_id + 1)]
            assert (abs(bc_data[0]) + abs(bc_data[1]) > 0)
            P = np.identity(self.n_dim)

            stencil, a = self.get_transversal_flux(cell_id1, face_id1, 0.0, 0.0)
            #stencil, a = self.transversal_fluxes[cell_id1][face_id1]
            if bc_data[2] == self.P12:  # Dirichlet / Neumann boundary
                tmp = np.linalg.inv(bc_data[0] * np.identity(self.n_dim) + bc_data[1] * P.dot(T1) / r1)
                T = T1.dot(tmp) / r1
                if bc_data[0] != 0.0:
                    a = (np.identity(self.n_dim) - bc_data[1] / r1 * (T1.dot(tmp)).dot(P)).dot(a)
                    a = np.array(np.hsplit(a, stencil.size))
                    non_zero_ind = (np.fabs(a) > self.tol).any(axis=(1, 2))
                    self.Ft1[cell_id1][face_id1] = ((cell_id2, face_id2), stencil[non_zero_ind], face.area * a[non_zero_ind])
                    self.Ft2[cell_id1][face_id1] = ((cell_id2, face_id2), np.array([]), np.array([]))
                    self.Fharm[cell_id1][face_id1] = ((cell_id2, face_id2),
                                                      np.array([bound_id + self.mat_cells_tot + self.frac_cells_tot, cell_id1]),
                                                      face.area * np.array([T, -bc_data[0] * T]))
                else:
                    self.Ft1[cell_id1][face_id1] = ((cell_id2, face_id2), np.array([]), np.array([]))
                    self.Ft2[cell_id1][face_id1] = ((cell_id2, face_id2), np.array([]), np.array([]))
                    self.Fharm[cell_id1][face_id1] = ((cell_id2, face_id2),
                                                      np.array([bound_id + self.mat_cells_tot + self.frac_cells_tot]),
                                                      face.area * np.array([T]))
            else:  # Roller boundary
                T1inv = np.linalg.inv(T1)
                tmp = np.outer(n, n) / n.dot(T1inv).dot(n)
                a = tmp.dot(T1inv).dot(a)
                a = np.array(np.hsplit(a, stencil.size))
                non_zero_ind = (np.fabs(a) > self.tol).any(axis=(1, 2))
                self.Ft1[cell_id1][face_id1] = ((cell_id2, face_id2), stencil[non_zero_ind], face.area * a[non_zero_ind])
                self.Ft2[cell_id1][face_id1] = ((cell_id2, face_id2), np.array([]), np.array([]))
                self.Fharm[cell_id1][face_id1] = ((cell_id2, face_id2),
                                                  np.array([cell_id1]),
                                                  -face.area * np.array([tmp]) / r1)
            self.Fharm[cell_id1][face_id1][2][np.where(np.fabs(self.Fharm[cell_id1][face_id1][2]) < self.tol)] = 0.0
            self.Ft1[cell_id1][face_id1][2][np.where(np.fabs(self.Ft1[cell_id1][face_id1][2]) < self.tol)] = 0.0
        elif face.type == FType.MAT_TO_FRAC:
            face0 = self.faces[cell_id1][face.face_id1]

            cell = self.mat_cell_info_dict[cell_id1]
            t_face1 = face.centroid - cell.centroid
            n = face.n
            if np.inner(t_face1, n) < 0: n = -n
            stf1 = self.stf[cell.prop_id]

            cell2 = self.mat_cell_info_dict[face0.cell_id2]
            t_face2 = cell2.centroid - face.centroid
            stf2 = self.stf[cell2.prop_id]
            T1, G1 = self.get_normal_tangential_stiffness(stf1, n)
            T2, G2 = self.get_normal_tangential_stiffness(stf2, n)
            r1 = np.inner(t_face1, n)
            r2 = np.inner(t_face2, n)
            Tden = np.linalg.inv(r1 * T2 + r2 * T1)
            T = T1.dot(Tden).dot(T2)
            y1 = cell.centroid + r1 * n
            y2 = cell2.centroid - r2 * n
            Gt1 = T.dot(np.kron(np.identity(self.n_dim), face.centroid - y2)) + \
                r2 * T1.dot(Tden).dot(G2)
            Gt2 = T.T.dot(np.kron(np.identity(self.n_dim), y1 - face.centroid)) - \
                r1 * T2.dot(Tden).dot(G1)

            t_face1 = face.centroid - self.mat_cell_info_dict[cell_id1].centroid
            sign = self.get_fracture_sign(-t_face1)
            conn0, conn1, conn2 = self.Fharm[cell_id1][face.face_id1]
            pos = 0
            if cell_id2 not in conn1:
                conn1 = np.append(conn1, cell_id2)
                conn2 = np.vstack([conn2, -sign * face.area * T[np.newaxis,:,:]])
            else:
                pos = np.argwhere(conn1 == cell_id2)[0][0]
                conn2[pos] += -sign * face.area * T

            self.Fharm[cell_id1][face.face_id1] = (conn0, conn1, conn2)
            self.Fharm[cell_id1][face.face_id1][2][np.where(np.fabs(self.Fharm[cell_id1][face.face_id1][2]) < self.tol)] = 0.0

            data = self.transversal_fluxes[cell_id2][0]
            if cell_id1 not in self.Fcont1:
                self.Fcont1[cell_id1] = {}
            if cell_id1 not in self.Fcont2:
                self.Fcont2[cell_id1] = {}

            self.Fcont1[cell_id1][face.face_id1] = (conn1, data[0], sign * face.area * np.array(np.hsplit(Gt1.dot(data[1]), data[0].size)))
            self.Fcont2[cell_id1][face.face_id1] = (conn1, data[0], sign * face.area * np.array(np.hsplit(Gt2.dot(data[1]), data[0].size)))
            self.Fcont1[cell_id1][face.face_id1][2][np.where(np.fabs(self.Fcont1[cell_id1][face.face_id1][2]) < self.tol)] = 0.0
            self.Fcont2[cell_id1][face.face_id1][2][np.where(np.fabs(self.Fcont2[cell_id1][face.face_id1][2]) < self.tol)] = 0.0
    def calc_frictionless_fracture_connection(self, cell_id):
        face1 = self.faces[cell_id][4]
        if self.get_fracture_sign(face1.n) > 0.0:
            face1 = self.faces[cell_id][5]

        S = np.zeros((self.n_dim, self.n_dim))
        S[:self.n_dim-1] = null_space(np.array([face1.n])).T
        S[self.n_dim-1] = face1.n
        Sinv = np.linalg.inv(S)

        conn = self.mpsa_connections[face1.cell_id2][face1.face_id2]
        P = np.identity(self.n_dim) - np.outer(face1.n, face1.n)
        F = np.concatenate(conn[2].transpose(0, 2, 1)).T / face1.area
        Ftan = P.dot(F)
        pos = np.argwhere(conn[1] == cell_id)[0][0]

        Ftan = S.dot(Ftan).dot(np.kron(np.identity(conn[1].size), Sinv))
        Ftan[2, pos * self.n_dim:(pos + 1) * self.n_dim] = S.dot(face1.n)
        Ftan = Ftan.dot(np.kron(np.identity(conn[1].size), S))
        self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)] = 0.0

        #Ftan[:,pos*self.n_dim:(pos+1)*self.n_dim][2] += face1.area * np.array([0.0,0.0,1.0])
        return (cell_id, 0), conn[1], np.array(np.hsplit(Ftan,conn[1].size))
        #return (cell_id, 0), np.array([cell_id]), np.array([np.identity(self.n_dim)])
    def local_iterations(self, cell_id, Fn, x_new):
        Ft_prev = self.Ft_prev[cell_id]

        x = self.x_prev[cell_id]
        # local unknowns - tangential traction and lagrange multiplier
        x_loc = np.zeros(4)
        x_loc[:3] = Ft_prev + self.eps_t * (x_new - x)
        norm = np.linalg.norm(x_loc[:3])
        # RHS
        rhs = np.zeros(4)
        rhs[:3] = x_new - x - x_loc[3] * x_loc[:3] / norm - (x_loc[:3] - Ft_prev) / self.eps_t
        rhs[3] = norm + self.mu * Fn
        # local jacobian
        jac = np.zeros((4, 4))
        jac[:3, :3] = x_loc[3] / norm * (np.outer(x_loc[:3] / norm, x_loc[:3] / norm) - np.identity(3)) - np.identity(
            3) / self.eps_t
        jac[:3, 3] = -x_loc[:3] / norm
        jac[3, :3] = x_loc[:3] / norm

        for i in range(50):
            jac_pinv = np.linalg.inv(jac)
            x_loc -= jac_pinv.dot(rhs)
            norm = np.linalg.norm(x_loc[:3])
            diff = np.linalg.norm(jac_pinv.dot(rhs))
            # RHS
            rhs[:3] = x_new - x - x_loc[3] * x_loc[:3] / norm - (x_loc[:3] - Ft_prev) / self.eps_t
            rhs[3] = norm + self.mu * Fn
            jac[:3, :3] = x_loc[3] / norm * (
                    np.outer(x_loc[:3] / norm, x_loc[:3] / norm) - np.identity(3)) - np.identity(3) / self.eps_t
            jac[:3, 3] = -x_loc[:3] / norm
            jac[3, :3] = x_loc[:3] / norm

            if diff < 1.E-12: break

        return x_loc, jac
    def get_variables(self, ids, type):
        size = self.bcm_num
        if type == 'new': x = self.x_new
        elif type == 'iter': x = self.x_iter
        elif type == 'prev': x = self.x_prev
        x_new = np.zeros((ids.size, self.n_dim))
        for i, id in enumerate(ids):
            if id >= self.mat_cells_tot + self.frac_cells_tot:
                bound_id = id - self.mat_cells_tot - self.frac_cells_tot
                x_new[i] = self.bc_mech[size * bound_id + 3:size * (bound_id + 1)]
            else:
                x_new[i] = x[id]
        return x_new

    def get_friction_and_derivative(self, slip):
        if np.isscalar(slip):
            if slip < self.d:
                mu = self.mu - (self.mu - self.mu_min) / self.d * slip
                mu_dslip = - (self.mu - self.mu_min) / self.d
            else:
                mu = self.mu_min
                mu_dslip = 0.0
            return mu, mu_dslip
        else:
            mu = np.zeros(slip.size)
            mu_dslip = np.zeros(slip.size)
            id1 = slip < d
            mu[id1] = self.mu - (self.mu - self.mu) / self.d * slip[id1]
            mu_dslip[id1] = - (self.mu - self.mu) / self.d

            id2 = slip >= d
            mu[id2] = self.mu_min
            mu_dslip[id2] = 0.0
            return mu, mu_dslip
    def return_mapping(self, cell_id):
        face1 = self.faces[cell_id][4]
        if self.get_fracture_sign(face1.n) > 0.0:
            face1 = self.faces[cell_id][5]

        S = np.zeros((self.n_dim, self.n_dim))
        S[:self.n_dim-1] = null_space(np.array([face1.n])).T
        S[self.n_dim-1] = face1.n
        Sinv = np.linalg.inv(S)
        #sign = self.get_fracture_sign(face1.centroid-self.mat_cell_info_dict[face1.cell_id2].centroid)
        conn = self.mpsa_connections[face1.cell_id2][face1.face_id2]
        P = np.identity(self.n_dim) - np.outer(face1.n, face1.n)
        # Global decomposition
        F_coef = np.concatenate(conn[2].transpose(0, 2, 1)).T / face1.area
        Fn_coef = np.outer(face1.n, face1.n).dot(F_coef)
        Ft_coef = P.dot(F_coef)
        #dFndg = Fn_coef[:,pos*self.n_dim:(pos+1)*self.n_dim]
        # Values
        F = F_coef.dot(self.get_variables(conn[1], 'new').flatten())
        Fn_vec = Fn_coef.dot(self.get_variables(conn[1], 'new').flatten())
        Fn = max(0.0, face1.n.dot(F))
        Ft = P.dot(F)

        Ft_prev = self.Ft_prev[cell_id]
        Ft_iter = self.Ft_iter[cell_id]
        x_new = P.dot(self.x_new[cell_id])
        x_iter = P.dot(self.x_iter[cell_id])
        x = P.dot(self.x_prev[cell_id])
        eps_t = self.eps_t

        pos = np.argwhere(conn[1] == cell_id)[0][0]
        #adv_gap = np.array([0.001,0.0,0.0])
        #x_loc1, jac1 = self.local_iterations(cell_id, Fn, x_new + adv_gap)
        #tmp = np.tile((x_loc1 - x_loc)[np.newaxis, :].T, (1, 3)) / adv_gap
        #Ft_trial = self.Ft_prev[cell_id] + eps_t * P.dot(self.x_new[cell_id] - self.x_prev[cell_id])
        #norm_trial = np.linalg.norm(Ft_trial)
        #Phi_trial = np.linalg.norm(Ft_trial) + self.mu * Fn
        #Ft_simple = Ft_trial - Phi_trial * Ft_trial / norm_trial
        #jac_simple = eps_t * P - eps_t / norm_trial ** 2 * np.outer(Ft_trial, Ft_trial).dot(P) - Phi_trial / norm_trial * (eps_t * P - eps_t / norm_trial ** 2 * np.outer(Ft_trial, Ft_trial).dot(P))
        #####
        Ft_norm = np.linalg.norm(Ft)
        dFn_norm = Fn_vec.dot(Fn_coef) / Fn if Fn > 0 else Fn_vec.dot(Fn_coef) / 1.E-8
        dFt_norm = Ft.dot(Ft_coef) / Ft_norm
        gt_norm = np.linalg.norm(x_new)
        if gt_norm == 0.0:
            gt_norm = 0.0000001 * Ft_norm / eps_t
        dgt_norm = x_new.dot(P) / gt_norm
        #H = Ft_coef + self.mu / Ft_norm * (np.outer(Ft, dFn_norm) + Fn * (Ft_coef - np.outer(Ft, dFt_norm) / Ft_norm))
        #H = self.mu / gt_norm * np.outer(x_new, dFn_norm)# + Fn * (Ft_coef - np.outer(Ft, dFt_norm) / Ft_norm))

        # get frictions and its derivative
        slip = np.linalg.norm(x_new)
        mu, dmu = self.get_friction_and_derivative(slip)
        if slip > 0:
            dmu = dmu * x_new / slip
        else:
            dmu = -dmu * Ft / Ft_norm

        Ft_trial = Ft_iter + eps_t * (x_new - x_iter)
        Ft_trial_norm = np.linalg.norm(Ft_trial)
        assert(Ft_trial_norm != 0)
        dFt_trial_norm = Ft_trial.dot(eps_t * P) / Ft_trial_norm
        H = Ft_coef - mu / Ft_norm * np.outer(Ft, dFn_norm)
        H[:, pos * self.n_dim:(pos + 1) * self.n_dim] -= mu * Fn / Ft_trial_norm * (eps_t * P - np.outer(Ft_trial, dFt_trial_norm) / Ft_trial_norm)
        H[:, pos * self.n_dim:(pos + 1) * self.n_dim] -= np.outer(Ft_trial, dmu) / Ft_trial_norm * Fn
        # Local basis
        H = S.dot(H).dot(np.kron(np.identity(conn[1].size), Sinv))
        assert((np.fabs(H[2,:]) < 1.E-12).all())
        H[2, pos * self.n_dim:(pos + 1) * self.n_dim] = S.dot(face1.n)
        H = H.dot(np.kron(np.identity(conn[1].size), S))

        self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)] = Ft - mu * Fn * Ft_trial / Ft_trial_norm  # x_loc[:3]-Ft_next-Ft_prev - eps_t *(x_new-x)
        self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)] = S.dot(self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)])
        assert(np.fabs(self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)][2]) < 1.E-12)
        # Subtract what will be added while jacobian assembly in engine
        Ft_next = H.dot(self.get_variables(conn[1], 'new').flatten())
        self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)] -= Ft_next

        return (cell_id, 0), conn[1], np.array(np.hsplit(H,conn[1].size))
        #return (cell_id, 0), np.array([cell_id]), np.array(np.hsplit(H,1))
    def calc_mpsa_connections_all_cells(self):
        # Start code for Matrix-Matrix and Fracture-Matrix connections:
        start_time_module = time.time()
        print('Start calculation MPSA connection list for matrix-matrix and matrix-fracture (if present) connections...')

        #A = np.zeros((2, self.n_dim, self.n_dim * self.n_dim), dtype=np.float64)
        self.mpsa_connections = {}
        self.mpsa_connections_num = 0
        for cell_id1, faces in self.faces.items():
            if cell_id1 not in self.mpsa_connections:
                self.mpsa_connections[cell_id1] = {}
            if cell_id1 < self.mat_cells_tot:
                for face_id1, face in faces.items():
                    cell_id2 = face.cell_id2
                    face_id2 = face.face_id2
                    self.mpsa_connections[cell_id1][face_id1] = self.calc_mpsa_connection(cell_id1, face_id1, cell_id2, face_id2)
                    self.mpsa_connections_num += 1
            else:
                #self.mpsa_connections[cell_id1][0] = self.calc_frictionless_fracture_connection(cell_id1)
                self.mpsa_connections[cell_id1][0] = (cell_id1, 0), np.array([cell_id1]), np.array([np.identity(self.n_dim)])
                self.mpsa_connections_num += 1
        #self.mpsa_connections_num += self.frac_cells_tot
        return self.calc_mpsa_contact_connections_update()
        #return self.flatten_mpsa_connections()
    def calc_mpsa_connections_all_cells_new(self):
        # Start code for Matrix-Matrix and Fracture-Matrix connections:
        start_time_module = time.time()
        print('Start calculation MPSA connection list for matrix-matrix and matrix-fracture (if present) connections...')

        #A = np.zeros((2, self.n_dim, self.n_dim * self.n_dim), dtype=np.float64)
        self.mpsa_connections = {}
        self.mpsa_connections_num = 0
        for cell_id1, faces in self.faces.items():
            if cell_id1 not in self.mpsa_connections:
                self.mpsa_connections[cell_id1] = {}
            if cell_id1 < self.mat_cells_tot:
                for face_id1, face in faces.items():
                    cell_id2 = face.cell_id2
                    face_id2 = face.face_id2
                    self.calc_mpsa_connection_by_terms(cell_id1, face_id1, cell_id2, face_id2)
                    #self.mpsa_connections[cell_id1][face_id1] = self.calc_mpsa_connection(cell_id1, face_id1, cell_id2, face_id2)
                for face_id1, face in faces.items():
                    self.mpsa_connections[cell_id1][face_id1] = self.merge_connection(cell_id1, face_id1)
                    self.mpsa_connections_num += 1
            else:
                #self.mpsa_connections[cell_id1][0] = self.calc_frictionless_fracture_connection(cell_id1)
                self.mpsa_connections[cell_id1][0] = (cell_id1, 0), np.array([cell_id1]), np.array([np.identity(self.n_dim)])
                self.mpsa_connections_num += 1
        #self.mpsa_connections_num += self.frac_cells_tot
        return self.calc_mpsa_contact_connections_update()
        #return self.flatten_mpsa_connections()
    def calc_mpsa_contact_connections_update(self):
        for cell_id in self.frac_cell_info_dict.keys():
            face1 = self.faces[cell_id][4]
            if self.get_fracture_sign(face1.n) > 0.0:
                face1 = self.faces[cell_id][5]
            #sign = self.get_fracture_sign(face1.centroid - self.mat_cell_info_dict[face1.cell_id2].centroid)
            if self.ith_iter == 0:
                self.mpsa_connections[cell_id][0] = ((cell_id, 0), np.array([cell_id]), np.array([np.identity(self.n_dim)]))
                continue
            else:
                conn = self.mpsa_connections[face1.cell_id2][face1.face_id2]

            P = np.identity(self.n_dim) - np.outer(face1.n, face1.n)
            # Global decomposition
            F_coef = np.concatenate(conn[2].transpose(0, 2, 1)).T / face1.area
            Fn_coef = np.outer(face1.n, face1.n).dot(F_coef)
            Ft_coef = P.dot(F_coef)
            # Values
            F = F_coef.dot(self.get_variables(conn[1], 'new').flatten())
            Fn = max(0.0, face1.n.dot(F))
            Ft = P.dot(F)

            if self.ith_iter == 1:
                self.Ft_prev[cell_id] = Ft

            self.Ft_iter[cell_id] = P.dot(F_coef.dot(self.get_variables(conn[1], 'iter').flatten()))

            #assert(self.get_fracture_sign(face1.n) == np.sign((self.Ft_iter[cell_id] * P.dot(self.x_new[cell_id] - self.x_iter[cell_id]))[0]) or np.sign((self.Ft_iter[cell_id] * P.dot(self.x_new[cell_id] - self.x_iter[cell_id]))[0]) == 0)
            eps_t = self.eps_t
            Ft_trial = self.Ft_iter[cell_id] + eps_t * P.dot(self.x_new[cell_id] - self.x_iter[cell_id])
            mu,dmu = self.get_friction_and_derivative(np.linalg.norm(P.dot(self.x_new[cell_id])))
            Phi_trial = np.linalg.norm(Ft_trial) - mu * Fn
            if Phi_trial > 0:# or np.linalg.norm(self.x_prev[cell_id]) == 0.0:
                # status slide
                self.mpsa_connections[cell_id][0] = self.return_mapping(cell_id)
            else:
                # status stick
                pos = np.argwhere(conn[1] == cell_id)[0][0]
                #H = Ft_coef
                #H[:, pos * self.n_dim:(pos + 1) * self.n_dim] -= eps_t * P
                H = P + P
                Ft_next = H.dot(self.x_new[cell_id].flatten())
                self.f[self.n_dim * cell_id:self.n_dim * (cell_id + 1)] = -Ft_next+P.dot(self.x_new[cell_id] - self.x_iter[cell_id])+(Ft_trial - self.Ft_iter[cell_id]) / eps_t# + face1.area * jac.dot(self.x_new[cell_id])
                #H[:, pos * self.n_dim:(pos + 1) * self.n_dim][1] += face1.area * face1.n
                H[1] += face1.area * face1.n
                #self.mpsa_connections[cell_id][0] = ((cell_id, 0), conn[1], np.array(np.hsplit(H, conn[1].size)))
                self.mpsa_connections[cell_id][0] = ((cell_id, 0), np.array([cell_id]), np.array([H]))
                #return (cell_id, 0), np.array([cell_id]), np.array([np.identity(self.n_dim)])
        return self.flatten_mpsa_connections()
    def flatten_mpsa_connections(self):
        cell_m = []
        cell_p = []
        stencil = []
        offset = []
        trans = []
        accum_size = 0
        for cell_id, faces in self.mpsa_connections.items():
            for face_id, data in faces.items():
                cell_id1 = data[0][0]
                isBound = cell_id < self.mat_cells_tot and cell_id == cell_id1
                isFrac = cell_id >= self.mat_cells_tot
                cell_m.append(cell_id)
                if isBound:
                    cell_p.append(self.mat_cells_tot + self.frac_cells_tot + self.faces[cell_id][face_id].face_id2)
                else:
                    cell_p.append(cell_id1)

                offset.append(accum_size)
                st_size = len(data[1])
                accum_size += st_size
                stencil.extend(data[1])
                trans.extend(data[2].flatten())
        offset.append(accum_size)

        #print('Time to calculate MPSA connection list: {:f} [sec]'.format((time.time() - start_time_module)))
        print('\t#Mat-Mat MPSA connections found:   {:d}'.format(self.mpsa_connections_num))
        #print('\t#Mat-Frac MPSA connections found:  {:d}'.format(count_mat_frac_conn))
        print('------------------------------------------------\n')

        return cell_m, cell_p, stencil, offset, trans