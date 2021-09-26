from darts.models.physics.chemical_kinetics_evaluators import *
from darts.models.physics.physics_base import PhysicsBase


class ChemicalKin(PhysicsBase):
    """"
       Class to generate chemical equilibrium physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """

    def __init__(self, timer, components, n_points, min_p, max_p, min_z, max_z, kin_data, log_based=False,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        """"
           Initialize Chemical class.
           Arguments:
                - timer: time recording object
                - components: list of components in model
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z, max_z: minimum and maximum composition
                - platform: target simulation platform - 'cpu' (default) or 'gpu'
                - itor_type: 'multilinear' (default) or 'linear' interpolator type
                - itor_mode: 'adaptive' (default) or 'static' OBL parametrization
                - itor_precision: 'd' (default) - double precision or 's' - single precision for interpolation
        """
        super().__init__(cache)
        self.timer = timer.node["simulation"]
        self.components = components
        self.n_components = len(components)
        self.phases = ['gas', 'water']
        self.n_phases = len(self.phases)
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.max_z = max_z
        self.n_vars = self.n_components
        self.vars = ['pressure'] + [c + ' composition' for c in components[:-1]]
        self.n_phases = len(self.phases)
        self.n_rate_temp_ops = self.n_phases

        self.n_axes_points = index_vector([n_points] * self.n_vars)
        self.n_axes_min = value_vector([min_p] + [min_z] * (self.n_components - 1))
        self.n_axes_max = value_vector([max_p] + [max_z] * (self.n_components - 1))

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        self.n_ops = 3 * self.n_components + 1
        self.engine = eval("engine_nc_kin_%s%d" % (platform, self.n_components))()

        # ------------------------------------------------------
        # Start definition parameters physics
        # ------------------------------------------------------
        physics_type = 'kinetics'  # nothing, diffusion, kinetics, kin_diff
        bool_trans_upd = False

        min_comp = min_z * 10
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
        components_data = component_acc_flux_data(vec_pressure_range_k_values,
                                                  vec_thermo_equi_const_k_water, vec_thermo_equi_const_k_co2,
                                                  sca_k_caco3, sca_tolerance, sca_ref_pres,
                                                  sca_density_water_stc,
                                                  sca_compressibility_water, sca_density_gas_stc,
                                                  sca_compressibility_gas,
                                                  sca_density_solid_stc, sca_compressibility_solid,
                                                  vec_res_sat_mobile_phases,
                                                  vec_brooks_corey_exponents, vec_end_point_rel_perm,
                                                  vec_viscosity_mobile_phases, sca_transmissibility_exp,
                                                  self.n_components, min_comp, kin_data)
        # ------------------------------------------------------
        # End definition parameters physics
        # ------------------------------------------------------

        # Initialize main evaluator
        self.acc_flux_etor = component_acc_flux_etor(components_data, bool_trans_upd, physics_type, log_based)
        self.acc_flux_etor_well = component_acc_flux_etor(components_data, bool_trans_upd, physics_type, log_based)

        self.acc_flux_itor = self.create_interpolator(self.acc_flux_etor, self.n_vars, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform, algorithm=itor_type, mode=itor_mode,
                                                      precision=itor_precision)

        self.acc_flux_itor_well = self.create_interpolator(self.acc_flux_etor_well, self.n_vars, self.n_ops,
                                                           self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                           platform=platform, algorithm=itor_type, mode=itor_mode,
                                                           precision=itor_precision)

        # create rate operators evaluator
        self.rate_etor = chemical_rate_evaluator(components_data, bool_trans_upd, physics_type, log_based)

        # interpolator platform is 'cpu' since rates are always computed on cpu
        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.n_rate_temp_ops,
                                                  self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                  platform='cpu', algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)

        # set up timers
        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.acc_flux_itor_well, 'well interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        # (vapor/gas and liquid/aqueous)
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells (vapor/gas and liquid/aqueous):
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            # w.init_rate_parameters(self.n_components + 1, self.phases, self.rate_itor)
            w.init_rate_parameters(self.n_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_temperature: uniform composition setting
        """
        assert isinstance(mesh, conn_mesh)
        assert len(uniform_composition) == self.n_components - 1
        nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]
