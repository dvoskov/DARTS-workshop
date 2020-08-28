# For the class defintions here, only numpy is used:
import numpy as np
from enum import Enum
# ------------------------------------------------------------
# Start matrix geometrical elements here: 3D objects
# Currently supported matrix
# ------------------------------------------------------------
""""
    Some definitions:
        - Nodes:    Vertices or points
        - Cells:    Control volumes
        - Face:     Sides of the control volume
        
    Most of the calculations regarding subdividing control volumes into tetrahedrons is taken from this paper:
    https://www.researchgate.net/publication/221561839_How_to_Subdivide_Pyramids_Prisms_and_Hexahedra_into_Tetrahedra
"""


# The following parent class contains all the definitions for the (currently) supported geometric objects for
# unstructured reservoir (typically when imported from GMSH, but should be generalizable to any type of mesh):
class ControlVolume:
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type):
        """
        Class constructor for the parents class ControlVolume
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        """
        # Initialize some object variables:
        self.volume = 0
        self.depth = 0
        self.centroid = 0
        self.nodes_to_cell = nodes_to_cell
        self.coord_nodes_to_cell = coord_nodes_to_cell
        self.geometry_type = geometry_type
        self.nodes_to_faces = {}

        # This might be new for people not familiar to OOP. It is possible to call class methods which are defined below
        # as (and can be overloaded by child class) during the construction of a class:
        self.calculate_centroid()  # Class method which calculates the centroid of the control volume
        self.calculate_depth()  # Class method which calculates the depth (center) of the control volume
        self.calculate_nodes_to_face()  # Class method which finds the array containing the nodes of each face of the CV
        self.calculate_volume()  # Class method which calculates the volume of the CV

    def calculate_centroid(self):
        """
        Class method that calculates the centroid of the control volume (just the arithmic mean of the nodes coordinates
        :return:
        """
        self.centroid = np.average(self.coord_nodes_to_cell, axis=0)
        return 0

    def calculate_depth(self):
        """
        Class method which calculates the depth of the particular control volume (at the  center of the volume)
        :return:
        """
        self.depth = np.abs(self.centroid[2])  # The class method assumes here that the third coordinate is the depth!
        return 0

    def calculate_nodes_to_face(self):
        """
        Virtual class method for finding the nodes to each face (overloaded by child classes, see their implementation)
        :return:
        """
        pass

    def calculate_volume(self):
        """
        Virtual class method for calculating the volume (overloaded by child classes, see their implementation)
        :return:
        """
        pass

    @staticmethod
    def compute_volume_tetrahedron(node_coord):
        """
        Static method which computes the volume of a tetrahedron based on the coordinates of the cell
        :param node_coord:
        :return: volume of the tetrahedron
        """
        # Even new(er) and faster way:
        vec_edge_1 = node_coord[0] - node_coord[3]
        vec_edge_2 = node_coord[1] - node_coord[3]
        vec_edge_3 = node_coord[2] - node_coord[3]

        # Using determinants to calculate volume of tetrahedron: https://en.wikipedia.org/wiki/Tetrahedron#Volume
        volume_tetra = np.linalg.norm(np.dot(vec_edge_1, np.cross(vec_edge_2, vec_edge_3))) / 6
        return volume_tetra

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Virtual class method for finding matrix or fracture intersection (overloaded by child classes, see their
        implementation)
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return:
        """
        pass


class Hexahedron(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the child class Hexahedron
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param permeability: permeability of control volume [mD]
        """
        # Call parent class constructor:
        super(Hexahedron, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Top and bottom faces (rectangles)
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[4], self.nodes_to_cell[5]]  # Bottom hexahedron
        self.nodes_to_faces[1] = [self.nodes_to_cell[2], self.nodes_to_cell[3],
                                  self.nodes_to_cell[6], self.nodes_to_cell[7]]  # Top hexahedron

        # Side faces (rectangles)
        self.nodes_to_faces[2] = [self.nodes_to_cell[4], self.nodes_to_cell[5],
                                  self.nodes_to_cell[6], self.nodes_to_cell[7]]  # Front hexahedron
        self.nodes_to_faces[3] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2], self.nodes_to_cell[3]]  # Back hexahedron
        self.nodes_to_faces[4] = [self.nodes_to_cell[0], self.nodes_to_cell[3],
                                  self.nodes_to_cell[4], self.nodes_to_cell[7]]  # Side hexahedron
        self.nodes_to_faces[5] = [self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[5], self.nodes_to_cell[6]]  # Side hexahedron
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split the hexahedron into five-tetrahedrons (see paper mentioned above for definitions and method):
        # Determine array with five-possible tetrahedrons (entries of array are nodes that belong to the CV):
        nodes_array_tetras = np.array([[4, 5, 1, 6],
                                       [4, 1, 0, 3],
                                       [4, 6, 3, 7],
                                       [1, 6, 3, 2],
                                       [4, 1, 6, 3]])

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3))

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            # Calculate volume for current tetra and add to total volume:
            self.volume = self.volume + Hexahedron.compute_volume_tetrahedron(local_coord)
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return:
        """
        # For hexahedron (8-node), four nodes make up interface:
        intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                 set(cells_to_node[nodes_to_face[1]]),
                                                 set(cells_to_node[nodes_to_face[2]]),
                                                 set(cells_to_node[nodes_to_face[3]]))
        return intsect_cells_to_face


class Wedge(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the child class Wedge
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param permeability: permeability of control volume [mD]
        """
        # Call parent class constructor:
        super(Wedge, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Add permeability to object variables:
        self.permeability = permeability  # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Top and bottom faces (triangles)
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2]]  # Bottom wedge
        self.nodes_to_faces[1] = [self.nodes_to_cell[3], self.nodes_to_cell[4],
                                  self.nodes_to_cell[5]]  # Top wedge

        # Side faces (rectangles)
        self.nodes_to_faces[2] = [self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[4], self.nodes_to_cell[5]]  # Front wedge
        self.nodes_to_faces[3] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[3], self.nodes_to_cell[4]]  # Side wedge
        self.nodes_to_faces[4] = [self.nodes_to_cell[0], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3], self.nodes_to_cell[5]]  # Side wedge
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split into three-tetrahedrons  (see paper mentioned above for definitions and method):
        # Determine array with five-possible tetrahedrons:
        nodes_array_tetras = np.array([[0, 1, 2, 4],
                                       [0, 3, 4, 5],
                                       [0, 1, 2, 5]])

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3))

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            # Calculate volume for current tetra and add to total volume:
            self.volume = self.volume + Wedge.compute_volume_tetrahedron(local_coord)
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return: the intersecting cells at the (inter)face that's investigated (if 2 then interface is a connection
                 between two control volumes)
        """
        # Determine what face we are looping over, since wedge has two types of faces: quad (sides)
        # and triangle (top/bottom):
        intsect_cells_to_face = set()
        if len(nodes_to_face) == 4:
            # Face has four nodes, therefore quad:
            intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                     set(cells_to_node[nodes_to_face[1]]),
                                                     set(cells_to_node[nodes_to_face[2]]),
                                                     set(cells_to_node[nodes_to_face[3]]))
        elif len(nodes_to_face) == 3:
            # Face has three nodes, therefore triangle:
            intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                     set(cells_to_node[nodes_to_face[1]]),
                                                     set(cells_to_node[nodes_to_face[2]]))
        return intsect_cells_to_face


class Pyramid(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the child class Pyramid
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param permeability: permeability of control volume [mD]
        """
        # Call parent class constructor:
        super(Pyramid, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Add permeability to object variables:
        self.permeability = permeability     # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        # Bottom faces (Quadrangle)
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2], self.nodes_to_cell[3]]  # Bottom Quadrangle

        # Side faces (Triangle)
        self.nodes_to_faces[1] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[4]]  # Top wedge
        self.nodes_to_faces[2] = [self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[4]]  # Top wedge
        self.nodes_to_faces[3] = [self.nodes_to_cell[2], self.nodes_to_cell[3],
                                  self.nodes_to_cell[4]]  # Top wedge
        self.nodes_to_faces[4] = [self.nodes_to_cell[3], self.nodes_to_cell[0],
                                  self.nodes_to_cell[4]]  # Top wedge
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split into two-tetrahedrons (see paper mentioned above for definitions and method):
        self.volume = 0
        # I think he determines the length of the edges in order to find the best shaped tetrahedrons (he uses the
        # resulting meshing in simulations, where orthogonality is important). For volume calculations the ordering
        # should not matter if I recall correct. (can always revert back changes to previous version!)
        nodes_array_tetras = np.array([[1, 2, 3, 4],
                                       [1, 3, 0, 4]])

        # Loop over all tetrahedrons:
        for jj, ith_tetra in enumerate(nodes_array_tetras):
            # Assign local coordinates:
            local_coord = np.zeros((4, 3))

            # Loop over local coordinates:
            for ii, ith_coord in enumerate(ith_tetra):
                # Append coordinates to local system:
                local_coord[ii] = self.coord_nodes_to_cell[ith_coord]

            # Calculate volume for current tetra and add to total volume:
            self.volume = self.volume + Pyramid.compute_volume_tetrahedron(local_coord)
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return: the intersecting cells at the (inter)face that's investigated (if 2 then interface is a connection
                 between two control volumes)
        """
        # Determine what face we are looping over, since Pyramid has two types
        # of faces: quad (bottom) and triangle (sides):
        intsect_cells_to_face = set()
        if len(nodes_to_face) == 4:
            # Face has four nodes, therefore quad:
            intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                     set(cells_to_node[nodes_to_face[1]]),
                                                     set(cells_to_node[nodes_to_face[2]]),
                                                     set(cells_to_node[nodes_to_face[3]]))
        elif len(nodes_to_face) == 3:
            # Face has three nodes, therefore triangle:
            intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                     set(cells_to_node[nodes_to_face[1]]),
                                                     set(cells_to_node[nodes_to_face[2]]))
        return intsect_cells_to_face


class Tetrahedron(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, permeability, prop_id = -1):
        """
        Class constructor for the child class Tetrahedron
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param permeability: permeability of control volume [mD]
        """
        # Call parent class constructor:
        super(Tetrahedron, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Add permeability to object variables:
        self.permeability = permeability     # Can be scalar or vector with [Kx, Ky, Kz]
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[2]]  # Top triangle
        self.nodes_to_faces[1] = [self.nodes_to_cell[0], self.nodes_to_cell[1],
                                  self.nodes_to_cell[3]]  # Side triangle
        self.nodes_to_faces[2] = [self.nodes_to_cell[0], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3]]  # Side triangle
        self.nodes_to_faces[3] = [self.nodes_to_cell[1], self.nodes_to_cell[2],
                                  self.nodes_to_cell[3]]  # Side triangle
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Calculate area of tetrahedron shaped control volume:
        self.volume = Tetrahedron.compute_volume_tetrahedron(self.coord_nodes_to_cell)
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return: the intersecting cells at the (inter)face that's investigated (if 2 then interface is a connection
                 between two control volumes)
        """
        # Face has three nodes, therefore triangle:
        intsect_cells_to_face = set.intersection(set(cells_to_node[nodes_to_face[0]]),
                                                 set(cells_to_node[nodes_to_face[1]]),
                                                 set(cells_to_node[nodes_to_face[2]]))
        return intsect_cells_to_face

# ------------------------------------------------------------
# Start fracture geometrical elements here: 2D objects
# ------------------------------------------------------------


class Quadrangle(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, frac_aperture, prop_id = -1):
        """
        Class constructor for the child class Quadrangle
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param frac_aperture: fracture aperture of control volume [m]
        :param prop_id: identifier of corresponding property
        """
        # Call parent class constructor:
        super(Quadrangle, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Store aperture in object and calculate permeability according to parallel plate law:
        self.frac_aperture = frac_aperture
        self.permeability = 1 / 12 * (self.frac_aperture ** 2) * 1E15
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1]]  # Top side triangle
        self.nodes_to_faces[1] = [self.nodes_to_cell[1], self.nodes_to_cell[2]]  # Left side triangle
        self.nodes_to_faces[2] = [self.nodes_to_cell[2], self.nodes_to_cell[3]]  # Right side triangle
        self.nodes_to_faces[3] = [self.nodes_to_cell[3], self.nodes_to_cell[0]]  # Right side triangle
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        # Split in two triangles, and calculate area
        # Create two vectors and calculate half area of parallelogram
        vec_corner_1 = self.coord_nodes_to_cell[0] - self.coord_nodes_to_cell[1]
        vec_corner_2 = self.coord_nodes_to_cell[0] - self.coord_nodes_to_cell[3]
        vec_corner_3 = self.coord_nodes_to_cell[2] - self.coord_nodes_to_cell[1]
        vec_corner_4 = self.coord_nodes_to_cell[2] - self.coord_nodes_to_cell[3]

        # Calculate the area of the quadrangle by splitting it up into two triangles, then use cross-product,
        # see: http://www.maths.usyd.edu.au/u/MOW/vectors/vectors-11/v-11-7.html
        cell_area = 0.5 * (np.linalg.norm(np.cross(vec_corner_1, vec_corner_2))
                           + np.linalg.norm(np.cross(vec_corner_3, vec_corner_4)))
        self.volume = cell_area * 10**(-4)  # Pseudo thickness of based on fracture aperture [m]
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return: the intersecting cells at the (inter)face that's investigated (if 2 then interface is a connection
                 between two control volumes)
        """
        # Determine the amount of fractures that intersect on face nodes:
        intsect_cells_of_face = list(
            set.intersection(set(cells_to_node[nodes_to_face[0]]), set(cells_to_node[nodes_to_face[1]])))
        return intsect_cells_of_face


class Triangle(ControlVolume):
    def __init__(self, nodes_to_cell, coord_nodes_to_cell, geometry_type, frac_aperture, prop_id = -1):
        """
        Class constructor for the child class Triangle
        :param nodes_to_cell: array with all the nodes belonging the the control volume (CV)
        :param coord_nodes_to_cell: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :param geometry_type: geometry of the control volume (e.g. hexahedron, quadrangle)
        :param frac_aperture: fracture aperture of control volume [m]
        :param prop_id: identifier of corresponding property
        """
        # Call parent class constructor:
        super(Triangle, self).__init__(nodes_to_cell, coord_nodes_to_cell, geometry_type)

        # Store aperture in object and calculate permeability according to parallel plate law:
        self.frac_aperture = frac_aperture
        self.permeability = 1/12 * (self.frac_aperture ** 2) * 1E15
        self.prop_id = prop_id

    def calculate_nodes_to_face(self):
        """
        Class method which overloads parent method for finding the list of nodes belonging to each face of the CV
        :return:
        """
        # Store nodes belonging to each face of object:
        self.nodes_to_faces[0] = [self.nodes_to_cell[0], self.nodes_to_cell[1]]  # Top side triangle
        self.nodes_to_faces[1] = [self.nodes_to_cell[0], self.nodes_to_cell[2]]  # Left side triangle
        self.nodes_to_faces[2] = [self.nodes_to_cell[1], self.nodes_to_cell[2]]  # Right side triangle
        return 0

    def calculate_volume(self):
        """
        Class method which overloads parent method for calculating the volume of the particular CV
        :return:
        """
        vec_corner_1 = self.coord_nodes_to_cell[0] - self.coord_nodes_to_cell[1]
        vec_corner_2 = self.coord_nodes_to_cell[0] - self.coord_nodes_to_cell[2]

        # Calculate the area of the triangle using cross-product,
        # see: http://www.maths.usyd.edu.au/u/MOW/vectors/vectors-11/v-11-7.html
        cell_area = 0.5 * np.linalg.norm(np.cross(vec_corner_1, vec_corner_2))
        self.volume = cell_area * 10**(-4)  # Pseudo thickness of based on fracture aperture [m]
        return 0

    def find_intersections(self, cells_to_node, nodes_to_face):
        """
        Class method which overloads parent method for finding matrix or fracture intersections
        :param cells_to_node: dictionary containing all the cells belonging to each control volume node (vertex)
        :param nodes_to_face: dictionary with for each face of the CV the nodes (vertices) that belonging to it
        :return: the intersecting cells at the (inter)face that's investigated (if 2 then interface is a connection
                 between two control volumes)
        """
        # Determine the amount of fractures that intersect on face nodes:
        intsect_cells_of_face = list(
            set.intersection(set(cells_to_node[nodes_to_face[0]]), set(cells_to_node[nodes_to_face[1]])))
        return intsect_cells_of_face

class FType(Enum):
    MAT = 0
    BORDER = 1
    FRAC_TO_MAT = 2
    MAT_TO_FRAC = 3
    FRAC = 4
class Face:
    def __init__(self, cell_id1, face_id1, cell_id2, face_id2, pts, id, type, f_aper = 0, n = 0):
        """
        Class constructor for the parents class ControlVolume
        :param cell_id1: id of the first neighbouring cell
        :param face_id1: id of corresponding face belonging to the first cell
        :param cell_id2: id of the second neighbouring cell
        :param face_id2: id of corresponding face belonging to the second cell
        :param pts: array of points
        :param id: id of face
        :param type: FaceType of face
        """
        self.cell_id1 = cell_id1
        self.face_id1 = face_id1
        self.cell_id2 = cell_id2
        self.face_id2 = face_id2
        self.type = type

        self.pts = pts
        self.n_pts = pts.shape[0]
        if self.n_pts > 2:
            self.sort_pts_in_circ_order()
            self.calc_area_and_centroid()
        elif self.n_pts > 1:
            self.n = n / np.linalg.norm(n)
            self.centroid = np.average(self.pts, axis=0)
            self.area = f_aper * np.linalg.norm(self.pts[1] - self.pts[0])
        else:
            exit(-1)
    def sort_pts_in_circ_order(self):
        n = np.cross(self.pts[1] - self.pts[0], self.pts[2] - self.pts[0])
        self.n = n / np.linalg.norm(n)
        c = np.average(self.pts, axis=0)

        tx = self.pts[0] - c
        tx /= np.linalg.norm(tx)
        ty = np.cross(self.n, tx)
        assert(np.cross(tx, ty).dot(self.n) > 0.0)

        phi = np.zeros(self.n_pts)
        for i in range(self.n_pts):
            t = self.pts[i] - c
            t /= np.linalg.norm(t)
            x = t.dot(tx)
            y = t.dot(ty)
            phi[i] = np.arctan2(y, x) * 180 / np.pi
        inds = np.argsort(phi)
        self.pts = self.pts[inds]
    def calc_area_and_centroid(self):
        area = 0.0
        c = np.zeros(3)
        for i in range(self.n_pts - 2):
            cur_area = np.linalg.norm(np.cross(self.pts[i+1] - self.pts[0],
                                            self.pts[i+2] - self.pts[0])) / 2.0
            area += cur_area
            inds = [0, i+1, i+2]
            c += cur_area * np.average(self.pts[inds], axis=0)
        self.area = area
        self.centroid = c / area



