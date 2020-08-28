# For the class defintions here, only numpy is used:
import numpy as np


# Class definition for the TransCalculations class (holds methods for calculating the interface transmissibility for
# several connection types: matrix-matrix, matrix-fracture, and fracture-fracture. Discretization of the fractures is
# based on the following paper: https://doi.org/10.2118/79699-MS
class TransCalculations:
    # Define Darcy constant for changes to correct units:
    darcy_constant = 0.0085267146719160104986876640419948

    @staticmethod
    def compute_area(coord_nodes_to_face):
        """
        Static method which computes the area of a triangle or quadrilateral/quad face (from a matrix element!)
        :param coord_nodes_to_face: array with the (x,y,z) coordinates of the nodes belonging to this face
        :return area: area of the particular face
        """
        area = 0
        if len(coord_nodes_to_face) == 3:
            # Face is a triangle
            # Create two vectors and calculate half area of parallelogram
            vec_edge_1 = coord_nodes_to_face[0] - coord_nodes_to_face[1]
            vec_edge_2 = coord_nodes_to_face[0] - coord_nodes_to_face[2]
            area = 0.5 * np.linalg.norm(np.cross(vec_edge_1, vec_edge_2))

        elif len(coord_nodes_to_face) == 4:
            # Face is quadrangle
            # Split in two triangles, and calculate area
            # Create two vectors and calculate half area of parallelogram
            vec_edge_1 = coord_nodes_to_face[0] - coord_nodes_to_face[1]
            vec_edge_2 = coord_nodes_to_face[0] - coord_nodes_to_face[2]
            vec_edge_3 = coord_nodes_to_face[1] - coord_nodes_to_face[3]
            vec_edge_4 = coord_nodes_to_face[2] - coord_nodes_to_face[3]
            area = 0.5 * (np.linalg.norm(np.cross(vec_edge_1, vec_edge_2))
                          + np.linalg.norm(np.cross(vec_edge_3, vec_edge_4)))
        return area

    @staticmethod
    def projection_con(centroid_i, centroid_j, n_unit, centroid_int):
        """
        Static method which calculates the projection (necessary for non-orthogonal connections)
        :param centroid_i: centroid coordinates of cell i (x,y,z)
        :param centroid_j: centroid coordinates of cell j (x,y,z)
        :param n_unit: vector which is orthogonal on the interface and has unit length
        :param centroid_int: centroid of the interface between cell i and cell j (x,y,z)
        :return res: coordinate of the projection (most "orthogonal" point between two centroids of the  cells)
                        --> this point is used instead of the actual centroid of the interface in trans calculations
        """
        p = centroid_j - centroid_i  # vector between 2 centroids of CV
        t = np.dot((centroid_int - centroid_i), n_unit) / np.dot(p, n_unit)
        res = centroid_i + t * p
        return res

    @staticmethod
    def calc_trans_mat_mat(connection, mat_cell_info_dict, coord_nodes_to_face):
        """
        Static method which calculates the matrix-matrix transmissibility of the particular interface:
        :param connection: list with the two connections --> [connection_i, connection_j]
        :param mat_cell_info_dict: dictionary of all matrix cells and associated information
        :param coord_nodes_to_face: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :return trans_i_j: interface transmissibility (matrix-matrix)
        """
        # Permeability input of the cell, heterogeneity
        centroid_i = np.array(mat_cell_info_dict[connection[0]].centroid)
        centroid_j = np.array(mat_cell_info_dict[connection[1]].centroid)
        centroid_int = np.average(coord_nodes_to_face, axis=0)
        conn_area = TransCalculations.compute_area(coord_nodes_to_face)

        # Calculate Normal vector on face plane with cross product
        vec_edge_1 = coord_nodes_to_face[1] - coord_nodes_to_face[0]  # Vector from p1p0
        vec_edge_2 = coord_nodes_to_face[2] - coord_nodes_to_face[0]  # Vector from p1p0

        n = np.cross(vec_edge_1, vec_edge_2)      # Cross product gives perpendicular vector to plane
        n_unit = n / np.linalg.norm(n)            # Normalize to get outward pointing unit vector

        # Projection to get the point on the face when the two centroid's are connected to each other.
        projection = TransCalculations.projection_con(centroid_i, centroid_j, n_unit, centroid_int)

        # Calc Ti
        f_i = centroid_i - projection  # result is vector between centroid_i and projection (note: length of f_i is D_i)
        f_i_unit = f_i / np.linalg.norm(f_i)  # normalize resulting vector

        perm_i = mat_cell_info_dict[connection[0]].permeability  # get permeability from cell info

        # Calculate (average) based on f permeability
        k_dir = perm_i * f_i_unit                      # Directionally scaled permeability
        con_perm = np.linalg.norm(np.array(k_dir))     # Interface scaled permeability value

        # Calculate transmissibility
        trans_i = conn_area * con_perm * abs(np.dot(n_unit, f_i_unit)) / np.linalg.norm(f_i)

        # Same for Tj
        f_j = centroid_j - projection  # result is projection on that face
        f_j_unit = f_j / np.linalg.norm(f_j)  # normalize

        perm_j = mat_cell_info_dict[connection[1]].permeability
        k_dir = perm_j * f_j_unit
        con_perm = np.linalg.norm(np.array(k_dir))
        trans_j = conn_area * con_perm * abs(np.dot(n_unit, f_j_unit)) / np.linalg.norm(f_j)

        # Harmonic average
        trans_i_j = trans_i * trans_j / (trans_i + trans_j) * TransCalculations.darcy_constant
        thermal_i_j = conn_area / (np.linalg.norm(f_i) + np.linalg.norm(f_j))
        return trans_i_j, thermal_i_j

    @staticmethod
    def calc_trans_mat_frac(mat_element_id, frac_element_id, mat_cell_info_dict, frac_cell_info_dict,
                            coord_nodes_to_face):
        """
        Static method which calculates the matrix-fracture transmissibility of the particular interface:
        :param mat_element_id: holds the local matrix id matrix block concerned in the interface transmissibility
        :param frac_element_id: holds the local fracture id fracture block concerned in the interface transmissibility
        :param mat_cell_info_dict: dictionary with all the relevant information of all matrix blocks
        :param frac_cell_info_dict: dictionary with all the relevant information of all fracture blocks
        :param coord_nodes_to_face: array with the (x,y,z) coordinates of the nodes belonging to this control volume
        :return trans_mat_frac: interface transmissibility (matrix-fracture)
        """
        # Permeability input of the cell, heterogeneity
        # Centroid of matrix cell:
        centroid_mat = np.array(mat_cell_info_dict[mat_element_id].centroid)
        centroid_int = np.average(coord_nodes_to_face, axis=0)
        conn_area = TransCalculations.compute_area(coord_nodes_to_face)

        # Calculate Normal vector of face plane with cross product
        vec_edge_1 = coord_nodes_to_face[1] - coord_nodes_to_face[0]  # Vector from p1p0
        vec_edge_2 = coord_nodes_to_face[2] - coord_nodes_to_face[0]  # Vector from p1p0

        n = np.cross(vec_edge_1, vec_edge_2)    # Cross product gives perpendicular vector to plane
        n_unit = n / np.linalg.norm(n)          # Normalize to get outward pointing unit vector

        # Calculate "local" centroid of fracture cell, as half aperture of
        # interface centroid along interface normal vector:
        centroid_frac = 0
        if np.linalg.norm(np.abs(n_unit * frac_cell_info_dict[frac_element_id].frac_aperture + centroid_int)) \
                < np.linalg.norm(np.abs(centroid_mat - centroid_int)):
            # Meaning, going along the interface normal and going closer to matrix centroid,
            # take negative direction of unit vector:
            centroid_frac = np.array(frac_cell_info_dict[frac_element_id].centroid) \
                            - n_unit * 1 / 2 * frac_cell_info_dict[frac_element_id].frac_aperture

        elif np.linalg.norm(np.abs(n_unit * frac_cell_info_dict[frac_element_id].frac_aperture + centroid_int)) \
                > np.linalg.norm(np.abs(centroid_mat - centroid_int)):
            # Meaning, going along the interface normal and going further away from matrix centroid,
            # going in correct direction along unit vector:
            centroid_frac = np.array(frac_cell_info_dict[frac_element_id].centroid) \
                            + n_unit * 1 / 2 * frac_cell_info_dict[frac_element_id].frac_aperture

        # Projection to get the point on the face when the two centroid's are connected to each other.
        projection = TransCalculations.projection_con(centroid_mat, centroid_frac, n_unit, centroid_int)

        # Calc Ti
        f_mat = centroid_mat - projection  # result is vector between m_i and projection
        f_mat_unit = f_mat / np.linalg.norm(f_mat)  # normalize

        perm_mat = mat_cell_info_dict[mat_element_id].permeability  # get permeability from cell info
        # calculate (average) based on f permeability
        con_perm_mat = np.linalg.norm(np.array(perm_mat * f_mat_unit))

        # Calculate transmissability
        trans_mat = conn_area * con_perm_mat * abs(np.dot(n_unit, f_mat_unit)) / np.linalg.norm(f_mat)

        # Same for Tj
        f_frac = centroid_frac - projection  # result is projection on that face
        f_frac_unit = f_frac / np.linalg.norm(f_frac)  # normalize

        perm_frac = frac_cell_info_dict[frac_element_id].permeability
        con_perm_frac = np.linalg.norm(np.array(perm_frac * f_frac_unit))
        trans_frac = conn_area * con_perm_frac * abs(np.dot(n_unit, f_frac_unit)) / np.linalg.norm(f_frac)

        # Harmonic average
        trans_mat_frac = trans_mat * trans_frac / (trans_mat + trans_frac) * TransCalculations.darcy_constant
        thermal_mat_frac = conn_area / (np.linalg.norm(f_mat) + np.linalg.norm(f_frac))
        return trans_mat_frac, thermal_mat_frac

    @staticmethod
    def calc_trans_frac_frac(connect_array, temp_frac_elem, frac_cell_info_dict, coord_frac_nodes):
        """
        Static method which calculates the fracture-fracture transmissibility of the particular interface:
        :param connect_array: array which holds the two fractures currently investigated as intersection
        :param temp_frac_elem: array which holds all the other fractures except frac_i of the intersection
        :param frac_cell_info_dict: dictionary with all the relevant information of all fracture blocks
        :param coord_frac_nodes: array with the (x,y,z) coordinates of the nodes belonging to this fracture face
        :return trans_i_j: interface transmissibility (fracture-fracture using STAR-DELTA transform. see paper above)
        """
        frac_elem_i = connect_array[0]
        frac_elem_j = connect_array[1]

        # Calculate area for intersection:
        # In 3D ({'triangle','4-node quad'}), intersection is line, so need
        # one additional dimension:
        dist_vec_fracs = coord_frac_nodes[0] - coord_frac_nodes[1]
        length_frac_intersection = np.linalg.norm(dist_vec_fracs)
        conn_area = length_frac_intersection * frac_cell_info_dict[frac_elem_i].frac_aperture

        # New way of computing fracture intersection centroid:
        # Need:
        #   - centroid of fracture element, denoted frac_cell_info_dict[frac_elem_i].centroid
        #   - node associated with interface, denoted coord_frac_nodes[0]
        #   - distance between other interface node, denoted coord_frac_nodes[0] - coord_frac_nodes[1]
        # Project (centroidFracCell- centroidFracInterface) onto interface vector
        # fracNodeCoord[0] - fracNodeCoord[1] using normal vector projection:
        # see https://en.wikipedia.org/wiki/Vector_projection
        # First for ithFrac:
        proj_frac_i_to_int = np.dot(frac_cell_info_dict[frac_elem_i].centroid - coord_frac_nodes[0], dist_vec_fracs) / \
                             np.dot(dist_vec_fracs, dist_vec_fracs)
        new_proj_cen_rac_i = coord_frac_nodes[0] + proj_frac_i_to_int * dist_vec_fracs

        # Then for jthFrac:
        proj_frac_j_to_int = np.dot(frac_cell_info_dict[frac_elem_j].centroid - coord_frac_nodes[0], dist_vec_fracs) / \
                             np.dot(dist_vec_fracs, dist_vec_fracs)
        new_proj_cen_rac_j = coord_frac_nodes[0] + proj_frac_j_to_int * dist_vec_fracs

        # Then for remaining fractures in intersection:
        array_temp_frac = np.array(temp_frac_elem)
        rem_frac_elem = array_temp_frac[array_temp_frac != frac_elem_j]
        proj_frac_rem_to_int = np.zeros(len(rem_frac_elem))
        new_proj_cen_rac_rem = np.zeros((3, len(rem_frac_elem)))
        dist_frac_rem = np.zeros(len(rem_frac_elem))
        perm_frac_rem = np.zeros(len(rem_frac_elem))
        alpha_frac_rem = np.zeros(len(rem_frac_elem))

        if rem_frac_elem.size:
            # Non-zero array, therefore more fractures than just the ith- and
            # jth-intersection:
            for ii, ith_rem_frac in enumerate(rem_frac_elem):
                # Compute it's projected centroid:
                proj_frac_rem_to_int[ii] = np.dot(frac_cell_info_dict[ith_rem_frac].centroid - coord_frac_nodes[0],
                                                  dist_vec_fracs) / np.dot(dist_vec_fracs, dist_vec_fracs)
                new_proj_cen_rac_rem[:, ii] = coord_frac_nodes[0] + proj_frac_rem_to_int[ii] * dist_vec_fracs

        # Centroid of fracture intersection is average of individual projected centroids:
        centroid_int = (new_proj_cen_rac_i + new_proj_cen_rac_j + np.sum(new_proj_cen_rac_rem, axis=1)) / \
                       (2 + len(rem_frac_elem))

        # Calcualte distance, permeability and hereby alpha parameters of fractures:
        # Distance main fracture element under inspection (ithElement):
        dist_frac_i = np.linalg.norm(frac_cell_info_dict[frac_elem_i].centroid - centroid_int)
        alpha_frac_i = conn_area * frac_cell_info_dict[frac_elem_i].permeability / dist_frac_i

        # If fracture has directional permeability use:
        # permFrac_i    = np.linalg.norm(cellFracInfoDict[ithFracElem]['perm'])

        # Distance remaining fractures in particular intersection, first: jthElement
        dist_frac_j = np.linalg.norm(frac_cell_info_dict[frac_elem_j].centroid - centroid_int)
        alpha_frac_j = conn_area * frac_cell_info_dict[frac_elem_j].permeability / dist_frac_j

        # If fracture has directional permeability use:
        # permFrac_j    = np.linalg.norm(cellFracInfoDict[jthFracElem]['perm'])

        # Compute distances for remaining fracture elements intersection interface:
        if rem_frac_elem.size:
            # Non-zero array, therefore more fractures than just the ith- and
            # jth-intersection:
            for ii, ith_rem_frac in enumerate(rem_frac_elem):
                # Compute it's distance:
                dist_frac_rem[ii] = np.linalg.norm(frac_cell_info_dict[ith_rem_frac].centroid - centroid_int)
                perm_frac_rem[ii] = frac_cell_info_dict[ith_rem_frac].permeability
                # If fracture has directional permeability use:
                # permFrac_others[ii]  = np.linalg.norm(cellFracInfoDict[ithRemFrac]['perm'])

            # Compute alpha parameter (see Karimi-Fard et al. 2003) for remianing/other fractures:
            alpha_frac_rem = conn_area * perm_frac_rem / dist_frac_rem

        # Calculate Tij with star-delta transformation:
        trans_i_j = TransCalculations.darcy_constant * (alpha_frac_i * alpha_frac_j) / \
                    (alpha_frac_i + alpha_frac_j + np.sum(alpha_frac_rem))
        thermal_i_j = conn_area / (dist_frac_i + dist_frac_j + np.sum(dist_frac_rem))
        return trans_i_j, thermal_i_j
