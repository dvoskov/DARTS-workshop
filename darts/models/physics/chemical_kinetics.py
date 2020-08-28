import numpy as np
from darts.engines import *
from darts.physics import *
from darts.models.physics.chemical_kinetics_evaluators import component_acc_flux_etor, component_acc_flux_data, chemical_rate_evaluator


# Define our own operator evaluator class
class ChemicalKin:
    def __init__(self, timer, components, n_points, min_p, max_p, min_z, max_z, kin_data, log_based=False):

        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.max_z = max_z
        self.components = components
        self.n_components = len(components)
        self.n_vars = self.n_components
        self.vars = ['pressure'] + [c + ' composition' for c in components[:-1]]
        self.phases = ['gas', 'water']
        self.n_phases = len(self.phases)

        physics_type = 'kinetics'  # nothing, diffusion, kinetics, kin_diff
        bool_trans_upd = False

        # ------------------------------------------------------
        # Start definition parameters physics
        # ------------------------------------------------------
        min_comp = self.min_z * 10
        sca_tolerance = np.finfo(float).eps

        # Define K-values as function or pressure:
        vec_pressure_range_k_values = np.linspace(50, 140, 10, endpoint=True)
        vec_thermo_equi_const_k_water = np.array(
            [0.1080, 0.0945, 0.0849, 0.0779, 0.0726, 0.0684, 0.0651, 0.0624, 0.0602, 0.0584])
        vec_thermo_equi_const_k_co2 = np.array(
            [1149, 972, 845, 750, 676, 617, 569, 528, 494, 465])

        # Set also the equilibrium constant for the chemical equilibrium:
        sca_k_caco3 = 55.508 * 10 ** (0)

        # Some fluid related parameters:
        sca_ref_pres = 50
        sca_density_water_stc = 1
        sca_compressibility_water = 10 ** (-6) * 1
        sca_density_gas_stc = 1
        sca_compressibility_gas = 10 ** (-6) * 1
        sca_density_solid_stc = 1
        sca_compressibility_solid = 10 ** (-6) * 1

        # Residual saturation of each mobile(!) phase: [water, gas]
        sca_transmissibility_exp = 3
        vec_res_sat_mobile_phases = [0, 0]
        vec_brooks_corey_exponents = np.array([2, 2])
        vec_end_point_rel_perm = np.array([1, 1])
        vec_viscosity_mobile_phases = np.array([0.5, 0.1])

        # Construct data class for composition physics:
        components_data = component_acc_flux_data(vec_pressure_range_k_values,
                                                vec_thermo_equi_const_k_water, vec_thermo_equi_const_k_co2,
                                                sca_k_caco3, sca_tolerance, sca_ref_pres,
                                                sca_density_water_stc,
                                                sca_compressibility_water, sca_density_gas_stc, sca_compressibility_gas,
                                                sca_density_solid_stc, sca_compressibility_solid,
                                                vec_res_sat_mobile_phases,
                                                vec_brooks_corey_exponents, vec_end_point_rel_perm,
                                                vec_viscosity_mobile_phases, sca_transmissibility_exp,
                                                self.n_components, min_comp, kin_data)

        # ------------------------------------------------------
        # End definition parameters physics
        # ------------------------------------------------------

        # Name of interpolation method and engine used for this physics:
        if physics_type == 'nothing':
            # Basic engine without kinetics and transmultiplier:
            engine_name = eval("engine_nc_cpu%d" % self.n_components)
            self.n_ops = 2 * self.n_components  #acc, flux
        elif physics_type == 'diffusion':
            # Basic engine without kinetics and transmultiplier:
            engine_name = eval("engine_nc_dif_cpu%d" % self.n_components)
            self.n_ops = 3 * self.n_components  #acc, flux, diff
        elif physics_type == 'kinetics':
            # Engine with kinetics:
            engine_name = eval("engine_nc_kin_cpu%d" % self.n_components)
            self.n_ops = 3 * self.n_components + 1  #acc, flux, kin + 1 x poro
        elif physics_type == 'kin_diff':
            # Engine with kinetics:
            engine_name = eval("engine_nc_kin_dif_cpu%d" % self.n_components)
            self.n_ops = 4 * self.n_components + 1   #acc, flux, kin, diff + 1 x poro

        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_components, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_components, self.n_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_components, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_components, self.n_phases))

        # Initialize main evaluator
        self.acc_flux_etor = component_acc_flux_etor(components_data, bool_trans_upd, physics_type, log_based)

        # # Some code here to debug:
        # solid_composition = 0.3
        # self.acc_flux_etor.vec_k_values = [0.1, 2.5]
        # normalized_composition = np.array([0.01, 0.67, 0.01, 0.01]) / (1 - solid_composition)
        # # normalized_composition = np.array([0.785685, 0.0085694, 0.10287, 0.10287])
        #
        # # Calculat phase-split with Newton:
        # non_lin_unkwns, sca_iter_counter = self.acc_flux_etor.two_phase_flash_newton(normalized_composition)
        # phase_fractions = np.array([(1 - non_lin_unkwns[6]) * (1 - solid_composition),
        #                             non_lin_unkwns[6] * (1 - solid_composition),
        #                             solid_composition])
        #
        # # Calculate phase-split with bi-section algo:
        # self.acc_flux_etor.vec_k_values = np.array([0.1, 2.5, 0.00001, 0.00001])
        # non_lin_unkwns_bi, sca_iter_counter_bi = self.acc_flux_etor.two_phase_flash_full_sys(normalized_composition)
        #
        # # Get three-phase fractions back from re-normalization:
        # phase_fractions_bi = np.array([(1 - non_lin_unkwns_bi[6]) * (1 - solid_composition),
        #                                non_lin_unkwns_bi[6] * (1 - solid_composition),
        #                                solid_composition])

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
        self.rate_etor = chemical_rate_evaluator(components_data, bool_trans_upd, physics_type, log_based)
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
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor, self.acc_flux_etor.axis_points,
                               self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):

        assert isinstance(mesh, conn_mesh)
        assert len(uniform_composition) == self.n_components - 1

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]

    def set_diffusion_boundary_cond(self, mesh, uniform_pressure, left_res_blocks, left_comp, right_res_blocks,
                                    right_comp):
        # TODO: Write code here which will fill composition based on left or right domain:
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)

        # Formula for index: c + 0*(nc - 1), c + 1*(nc - 1), c + 2*(nc - 1), ..., c + nb*(nc - 1)
        left_idxs = np.array(left_res_blocks) * (self.n_components - 1)
        right_idxs = np.array(right_res_blocks) * (self.n_components - 1)

        # Loop over components and fill empty vector for initial composition:
        for c in range(self.n_components - 1):
            composition[(left_idxs + c)] = left_comp[c]
            composition[(right_idxs + c)] = right_comp[c]

        return 0

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)
        assert len(composition_bc) == self.n_components - 1

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]
