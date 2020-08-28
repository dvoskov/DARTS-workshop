from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *


class Elasticity:
    """"
       Class to generate deadoil physics for poromechanical simulation, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, n_points, max_u, n_dim):
        """"
           Initialize DeadOil class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
                - n_dim: space dimension
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.n_components = 0
        self.n_dim = n_dim
        self.n_vars = self.n_dim + self.n_components
        self.vars = ['displacement']

        grav = 0
        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        #if grav:
        #    engine_name = eval("engine_nc_pm_cpu%d_%d" % (self.n_components, self.n_dim))
        #    acc_flux_etor_name = elasticity_flux_evaluator
        #    self.n_ops = 0
        #else:
        #    exit(-1)
        engine_name = eval("engine_elasticity_cpu%d" % (self.n_dim))
        acc_flux_etor_name = elasticity_flux_evaluator
        self.n_ops = self.n_dim

        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_ops))
        #acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))

        # read keywords from physics file
        #pvdo = get_table_keyword(physics_filename, 'PVDO')
        #swof = get_table_keyword(physics_filename, 'SWOF')
        #pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
        #dens = get_table_keyword(physics_filename, 'DENSITY')[0]
        #rock = get_table_keyword(physics_filename, 'ROCK')

        # create property evaluators
        self.density = 2.E+3
        self.el_dens_ev = elasticity_string_density_evaluator(self.density)

        # create accumulation and flux operators evaluator
        self.acc_flux_etor = acc_flux_etor_name(self.el_dens_ev)
        self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points, n_points, n_points]),
                                    value_vector([-max_u, -max_u, -max_u]), value_vector([max_u, max_u, max_u]))

        self.timer.node["jacobian assembly"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
        self.acc_flux_itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])
        # set up timers
        self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        # create engine according to physics selected
        self.engine = engine_name()

    def init_wells(self, wells):
        return 0
    def set_uniform_initial_conditions(self, mesh, uniform_displacement: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_displacement: uniform displacement setting
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        assert(self.n_dim == len(uniform_displacement))

        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacement[i]

