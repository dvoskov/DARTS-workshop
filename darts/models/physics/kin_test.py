import numpy as np
from darts.engines import *
from darts.physics import *
from darts.models.physics.kin_test_eval import component_acc_flux_etor, chemical_rate_evaluator


# Define our own operator evaluator class
class ChemicalKin:
    def __init__(self, timer, components, n_points, min_p, max_p, min_z, max_z, log_based, kin_data):

        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.max_z = max_z
        self.components = components
        self.nr_components = len(components)
        self.phases = ['gas', 'water']
        self.nr_phases = len(self.phases)

        # ------------------------------------------------------
        # End definition parameters physics
        # ------------------------------------------------------

        # Name of interpolation method and engine used for this physics:
        # Engine with kinetics:
        engine_name = eval("engine_nc_kin_cpu%d" % self.nr_components)
        self.nr_ops = 3 * self.nr_components + 1
        # self.nr_ops = 2 * self.nr_components + 2

        # engine_name = eval("engine_nc_dif_cpu%d" % self.nr_components)
        # self.nr_ops = 3 * self.nr_components

        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.nr_components, self.nr_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.nr_components, self.nr_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.nr_components, self.nr_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.nr_components, self.nr_phases))

        # Initialize main evaluator
        num_property_evaluated = 0
        self.acc_flux_etor = component_acc_flux_etor(kin_data, log_based, self.nr_components, self.min_z)

        # Initialize table entries (nr of points, axis min, and axis max):
        # nr_of_points for [pres, comp1, ..., compN-1]:
        self.acc_flux_etor.axis_points = index_vector([self.n_points, self.n_points, self.n_points, self.n_points, self.n_points])
        # axis_min for [pres, comp1, ..., compN-1]:
        self.acc_flux_etor.axis_min = value_vector([self.min_p, min_z, min_z, min_z, min_z])
        # axis_max for [pres, comp1, ..., compN-1]
        self.acc_flux_etor.axis_max = value_vector([self.max_p, max_z, max_z, max_z, max_z])

        # Create actual accumulation and flux interpolator:
        try:
            self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, self.acc_flux_etor.axis_points,
                                                    self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)
        except RuntimeError:
            self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, self.acc_flux_etor.axis_points,
                                                         self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

        # set up timers
        self.timer.node["jacobian assembly"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
        self.acc_flux_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])

        # Create rate evaluator and interpolator:
        self.rate_etor = chemical_rate_evaluator()
        try:
            self.rate_itor = rate_interpolator_name(self.rate_etor, self.acc_flux_etor.axis_points,
                                                    self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)
        except RuntimeError:
            self.rate_itor = rate_interpolator_name_long(self.rate_etor, self.acc_flux_etor.axis_points,
                                                         self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

        # set up timers
        self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        self.rate_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"])

        # create engine according to physics selected
        self.engine = engine_name()

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.nr_components,
                                                                               self.nr_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.nr_components,
                                                                               self.nr_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.nr_components,
                                                                     self.nr_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.nr_components,
                                                                     self.nr_components,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor, self.acc_flux_etor.axis_points,
                               self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.nr_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):

        assert isinstance(mesh, conn_mesh)
        assert len(uniform_composition) == self.nr_components - 1

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.nr_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.nr_components - 1):
            composition[c::(self.nr_components - 1)] = uniform_composition[c]

    def set_diffusion_boundary_cond(self, mesh, uniform_pressure, left_res_blocks, left_comp, right_res_blocks,
                                    right_comp):
        # TODO: Write code here which will fill composition based on left or right domain:
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.nr_components - 1))
        composition = np.array(mesh.composition, copy=False)

        # Formula for index: c + 0*(nc - 1), c + 1*(nc - 1), c + 2*(nc - 1), ..., c + nb*(nc - 1)
        left_idxs = np.array(left_res_blocks) * (self.nr_components - 1)
        right_idxs = np.array(right_res_blocks) * (self.nr_components - 1)

        # Loop over components and fill empty vector for initial composition:
        for c in range(self.nr_components - 1):
            composition[(left_idxs + c)] = left_comp[c]
            composition[(right_idxs + c)] = right_comp[c]

        return 0

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)
        assert len(composition_bc) == self.nr_components - 1

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.nr_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]
