import numpy as np
from darts.engines import *
from darts.physics import *
from darts.models.physics.chemical_evaluators import element_acc_flux_etor, element_acc_flux_data, chemical_rate_evaluator


# Define our own operator evaluator class
class Chemical:
    def __init__(self, timer, n_elements, n_points, min_p, max_p, min_e, max_e):

        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_e = min_e
        self.max_e = max_e
        self.elements = n_elements
        self.n_elements = len(n_elements)
        self.n_vars = self.n_elements
        self.vars = ['pressure'] + [e + ' composition' for e in n_elements[:-1]]
        self.phases = ['gas', 'water']
        self.n_phases = len(self.phases)

        bool_trans_upd = True

        # ------------------------------------------------------
        # Start definition parameters physics
        # ------------------------------------------------------
        # Define rate-annihilation matrix E:
        mat_rate_annihilation = np.array([[1, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0],
                                          [0, 0, 1, 0, 1],
                                          [0, 0, 0, 1, 1]])

        min_comp = min_e * 10
        sca_tolerance = np.finfo(float).eps

        # Define K-values as function or pressure:
        vec_pressure_range_k_values = np.linspace(1, 500, 10, endpoint=True)
        vec_thermo_equi_const_k_water = np.array(
            [0.1080, 0.0945, 0.0849, 0.0779, 0.0726, 0.0684, 0.0651, 0.0624, 0.0602, 0.0584])
        vec_thermo_equi_const_k_co2 = np.array(
            [1149, 972, 845, 750, 676, 617, 569, 528, 494, 465])

        # Set also the equilibrium constant for the chemical equilibrium:
        sca_k_caco3 = 55.508 * 10 ** (0)

        # Some fluid related parameters (realistic densities give less fingerng pattern, probably reason is that for
        # same composition of fluid/solid mixture, S_solid = phase_frac / phase_dens is drastically different when
        # changing the densities: equal densities and low H2O means very high S_s and very low porosity and therefore
        # very hard to infiltrate rock (small pertubations in porosity have large effect)! whereas realistic densities
        # mean lower S_s for the same H2O means higher porosity and therefore more easy to infiltrate rock and therefore
        # pertubations have smaller effect on the fingering pattern.
        # Probably fingering pattern is function of densities --> solid saturation (due to trans_mult)
        # Probably fingering pattern is function of equilibrium constant --> check tomorrow!
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

        # Construct data class for elements physics:
        elements_data = element_acc_flux_data(mat_rate_annihilation, vec_pressure_range_k_values,
                                              vec_thermo_equi_const_k_water, vec_thermo_equi_const_k_co2,
                                              sca_k_caco3, sca_tolerance, sca_ref_pres,
                                              sca_density_water_stc,
                                              sca_compressibility_water, sca_density_gas_stc, sca_compressibility_gas,
                                              sca_density_solid_stc, sca_compressibility_solid,
                                              vec_res_sat_mobile_phases,
                                              vec_brooks_corey_exponents, vec_end_point_rel_perm,
                                              vec_viscosity_mobile_phases, sca_transmissibility_exp, min_comp)
        # ------------------------------------------------------
        # End definition parameters physics
        # ------------------------------------------------------

        # Name of interpolation method and engine used for this physics:
        engine_name = eval("engine_nc_cpu%d" % self.n_elements)
        self.n_ops = 2 * self.n_elements
        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_elements, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_elements, self.n_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_elements, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_elements, self.n_phases))

        # Initialize main evaluator
        self.acc_flux_etor = element_acc_flux_etor(elements_data, bool_trans_upd)

        # Initialize table entries (nr of points, axis min, and axis max):
        # nr_of_points for [pres, comp1, ..., compN-1]:
        self.acc_flux_etor.axis_points = index_vector([self.n_points, self.n_points, self.n_points])
        # axis_min for [pres, comp1, ..., compN-1]:
        self.acc_flux_etor.axis_min = value_vector([self.min_p, min_e, min_e])
        # axis_max for [pres, comp1, ..., compN-1]
        self.acc_flux_etor.axis_max = value_vector([self.max_p, max_e, max_e])

        # Create actual accumulation and flux interpolator:
        try:
            self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, self.acc_flux_etor.axis_points,
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
        self.rate_etor = chemical_rate_evaluator(elements_data, bool_trans_upd)
        try:
            self.rate_itor = rate_interpolator_name_long(self.rate_etor, self.acc_flux_etor.axis_points,
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
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_elements,
                                                                               self.n_elements, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_elements,
                                                                               self.n_elements, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_elements,
                                                                     self.n_elements,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_elements,
                                                                     self.n_elements,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor, self.acc_flux_etor.axis_points,
                               self.acc_flux_etor.axis_min, self.acc_flux_etor.axis_max)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_elements, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):

        assert isinstance(mesh, conn_mesh)
        assert len(uniform_composition) == self.n_elements - 1

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_elements - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_elements - 1):
            composition[c::(self.n_elements - 1)] = uniform_composition[c]

    def set_boundary_conditions(self, mesh, pressure_bc, composition_bc, boundary_cells):
        assert isinstance(mesh, conn_mesh)
        assert len(composition_bc) == self.n_elements - 1

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure[boundary_cells] = pressure_bc

        mesh.composition.resize(mesh.n_blocks * (self.n_elements - 1))
        composition = np.array(mesh.composition, copy=False)

        for c in range(self.n_elements - 1):
            composition[boundary_cells * 2 + c] = composition_bc[c]
