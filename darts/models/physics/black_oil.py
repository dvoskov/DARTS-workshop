from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from math import fabs
from darts.models.physics.saturation_initialization.sat_z import *

class BlackOil:
    """"
       Class to generate blackoil physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, n_points, min_p, max_p, min_z):
        """"
           Initialize BlackOil class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.n_components = 3
        self.n_vars = self.n_components
        self.phases = ['gas', 'oil', 'water']
        self.vars = ['pressure', 'gas composition', 'oil composition']
        self.n_phases = len(self.phases)

        # gravity is taken into account or not
        grav = 1
        try:
            scond = get_table_keyword(physics_filename, 'SCOND')[0]     # Read in standard condition setting from file
            if len(scond) > 2 and fabs(scond[2]) < 1e-5:                # Gravity on or off, default: on
                grav = 0
        except:
            grav = 1

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        # Different engines, accumulation_flux_operator_evaluators and number of operators for gravity on or off
        if grav:
            engine_name = eval("engine_nc_cg_cpu%d_%d" % (self.n_components, self.n_phases))
            acc_flux_etor_name = black_oil_acc_flux_capillary_evaluator
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.n_phases
        else:
            engine_name = eval("engine_nc_cpu%d" % self.n_components)
            acc_flux_etor_name = black_oil_acc_flux_evaluator
            self.n_ops = 2 * self.n_components

        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_phases))

        # read keywords from physics file
        pvto = get_table_keyword(physics_filename, 'PVTO')
        pvdg = get_table_keyword(physics_filename, 'PVDG')
        swof = get_table_keyword(physics_filename, 'SWOF')
        sgof = get_table_keyword(physics_filename, 'SGOF')
        rock = get_table_keyword(physics_filename, 'ROCK')
        pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
        dens = get_table_keyword(physics_filename, 'DENSITY')[0]

        surface_oil_dens = dens[0]
        surface_water_dens = dens[1]
        surface_gas_dens = dens[2]

        swof_well = []
        swof_well.append(value_vector([swof[0][0],  swof[0][1],  swof[0][2],  0.0]))
        swof_well.append(value_vector([swof[-1][0], swof[-1][1], swof[-1][2], 0.0]))

        sgof_well = []
        sgof_well.append(value_vector([sgof[0][0],  sgof[0][1],  sgof[0][2], 0.0]))
        sgof_well.append(value_vector([sgof[-1][0], sgof[-1][1], sgof[-1][2], 0.0]))

        # corey_pcow = value_vector([0.5, 0.12, 0.16, 0.2])  ## exponent, phase residual, oil residual, entry pressure
        # corey_pcgo = value_vector([0.8, 0., 0., 0.12])     ## exponent, phase residual, oil residual, entry pressure
        # corey_well = value_vector([0.5, 0.12, 0.16, 0.])   ## no capillary pressure in wells pd = 0

        # create property evaluators
        self.bo_bubble_pres_ev = black_oil_bubble_pressure_evaluator(pvto, surface_oil_dens, surface_gas_dens)
        self.bo_rs_ev = black_oil_rs_evaluator(pvto, self.bo_bubble_pres_ev, surface_oil_dens, surface_gas_dens)
        self.bo_xgo_ev = black_oil_xgo_evaluator(self.bo_rs_ev, surface_gas_dens, surface_oil_dens)
        self.bo_gas_dens_ev = dead_oil_table_density_evaluator(pvdg, surface_gas_dens)
        self.bo_water_dens_ev = dead_oil_string_density_evaluator(pvtw, surface_water_dens)
        self.bo_oil_dens_ev = black_oil_oil_density_evaluator(pvto, surface_oil_dens, self.bo_bubble_pres_ev,
                                                              self.bo_xgo_ev)
        self.bo_gas_visco_ev = dead_oil_table_viscosity_evaluator(pvdg)
        self.bo_water_visco_ev = dead_oil_string_viscosity_evaluator(pvtw)
        self.bo_oil_visco_ev = black_oil_oil_viscosity_evaluator(pvto, self.bo_bubble_pres_ev)
        self.bo_xcp_gas_ev = black_oil_xcp_gas_evaluator(self.bo_rs_ev, self.bo_bubble_pres_ev, self.bo_xgo_ev,
                                                         surface_gas_dens, surface_oil_dens)
        self.bo_water_sat_ev = black_oil_water_saturation_evaluator(self.bo_bubble_pres_ev, self.bo_xgo_ev,
                                                                    self.bo_water_dens_ev, self.bo_oil_dens_ev,
                                                                    self.bo_gas_dens_ev)
        self.bo_oil_sat_ev = black_oil_oil_saturation_evaluator(self.bo_bubble_pres_ev, self.bo_xgo_ev,
                                                                self.bo_water_dens_ev, self.bo_oil_dens_ev,
                                                                self.bo_gas_dens_ev)
        self.bo_gas_sat_ev = black_oil_gas_saturation_evaluator(self.bo_oil_sat_ev, self.bo_water_sat_ev)
        self.bo_krow_ev = table_phase2_relative_permeability_evaluator(self.bo_water_sat_ev, swof)
        self.bo_krog_ev = table_phase2_relative_permeability_evaluator(self.bo_gas_sat_ev, sgof)
        self.bo_oil_relperm_ev = black_oil_oil_relative_permeability_evaluator(self.bo_water_sat_ev, self.bo_gas_sat_ev,
                                                                               self.bo_krow_ev, self.bo_krog_ev,
                                                                               swof, sgof)
        self.bo_water_relperm_ev = table_phase1_relative_permeability_evaluator(self.bo_water_sat_ev, swof)
        self.bo_gas_relperm_ev = table_phase1_relative_permeability_evaluator(self.bo_gas_sat_ev, sgof)
        self.bo_pcow_ev = table_phase_capillary_pressure_evaluator(self.bo_water_sat_ev, swof)
        self.bo_pcgo_ev = table_phase_capillary_pressure_evaluator(self.bo_gas_sat_ev, sgof)
        self.rock_compaction_ev = rock_compaction_evaluator(rock)
        # rel perm for wells
        self.bo_krw_well_ev = table_phase1_relative_permeability_evaluator(self.bo_water_sat_ev, swof_well)
        self.bo_krg_well_ev = table_phase1_relative_permeability_evaluator(self.bo_gas_sat_ev, sgof_well)
        self.bo_krow_well_ev = table_phase2_relative_permeability_evaluator(self.bo_water_sat_ev, swof_well)
        self.bo_krog_well_ev = table_phase2_relative_permeability_evaluator(self.bo_gas_sat_ev, sgof_well)
        self.bo_kro_well_ev = black_oil_oil_relative_permeability_evaluator(self.bo_water_sat_ev, self.bo_gas_sat_ev,
                                                                               self.bo_krow_well_ev, self.bo_krog_well_ev,
                                                                               swof_well, sgof_well)

        # create accumulation and flux operators evaluator
        if grav:
            # read the pc table for wells
            self.bo_pcow_w_ev = table_phase_capillary_pressure_evaluator(self.bo_water_sat_ev, swof_well)
            self.bo_pcgo_w_ev = table_phase_capillary_pressure_evaluator(self.bo_gas_sat_ev, sgof_well)

            self.acc_flux_etor = acc_flux_etor_name(self.bo_bubble_pres_ev, self.bo_rs_ev, self.bo_xgo_ev,
                                                self.bo_oil_dens_ev, self.bo_oil_visco_ev, self.bo_oil_sat_ev,
                                                self.bo_oil_relperm_ev, self.bo_water_dens_ev, self.bo_water_sat_ev,
                                                self.bo_water_visco_ev, self.bo_water_relperm_ev, self.bo_gas_dens_ev,
                                                self.bo_gas_visco_ev, self.bo_gas_relperm_ev, self.bo_xcp_gas_ev,
                                                self.bo_krow_ev, self.bo_krog_ev, self.bo_pcow_ev, self.bo_pcgo_ev,
                                                self.rock_compaction_ev)
            self.acc_flux_w_etor = acc_flux_etor_name(self.bo_bubble_pres_ev, self.bo_rs_ev, self.bo_xgo_ev,
                                                self.bo_oil_dens_ev, self.bo_oil_visco_ev, self.bo_oil_sat_ev,
                                                self.bo_oil_relperm_ev, self.bo_water_dens_ev, self.bo_water_sat_ev,
                                                self.bo_water_visco_ev, self.bo_water_relperm_ev, self.bo_gas_dens_ev,
                                                self.bo_gas_visco_ev, self.bo_gas_relperm_ev, self.bo_xcp_gas_ev,
                                                self.bo_krow_ev, self.bo_krog_ev, self.bo_pcow_w_ev, self.bo_pcgo_w_ev,
                                                self.rock_compaction_ev)
        else:
            self.acc_flux_etor = acc_flux_etor_name(self.bo_bubble_pres_ev, self.bo_rs_ev, self.bo_xgo_ev,
                                                self.bo_oil_dens_ev, self.bo_oil_visco_ev, self.bo_oil_sat_ev,
                                                self.bo_oil_relperm_ev, self.bo_water_dens_ev, self.bo_water_sat_ev,
                                                self.bo_water_visco_ev, self.bo_water_relperm_ev, self.bo_gas_dens_ev,
                                                self.bo_gas_visco_ev, self.bo_gas_relperm_ev, self.bo_xcp_gas_ev,
                                                self.bo_krow_ev, self.bo_krog_ev, self.rock_compaction_ev)
            self.acc_flux_w_etor = acc_flux_etor_name(self.bo_bubble_pres_ev, self.bo_rs_ev, self.bo_xgo_ev,
                                                self.bo_oil_dens_ev, self.bo_oil_visco_ev, self.bo_oil_sat_ev,
                                                self.bo_oil_relperm_ev, self.bo_water_dens_ev, self.bo_water_sat_ev,
                                                self.bo_water_visco_ev, self.bo_water_relperm_ev, self.bo_gas_dens_ev,
                                                self.bo_gas_visco_ev, self.bo_gas_relperm_ev, self.bo_xcp_gas_ev,
                                                self.bo_krow_ev, self.bo_krog_ev, self.rock_compaction_ev)

        # create accumulation and flux operators interpolator
        # with adaptive uniform parametrization (accuracy is defined by 'n_points')
        # of compositional space (range is defined by 'min_p', 'max_p', 'min_z')
        try:
            # try first to create interpolator with 4-byte index type
            self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points] * self.n_components),
                                                    value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                    value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))
            self.acc_flux_w_itor = acc_flux_itor_name(self.acc_flux_w_etor, index_vector([n_points] * self.n_components),
                                                    value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                    value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))
        except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
            self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor,
                                                         index_vector([n_points] * self.n_components),
                                                         value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                         value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))
            self.acc_flux_w_itor = acc_flux_itor_name_long(self.acc_flux_w_etor,
                                                         index_vector([n_points] * self.n_components),
                                                         value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                         value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))

        # set up timers
        self.timer.node["jacobian assembly"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
        self.acc_flux_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux w interpolation"] = timer_node()
        self.acc_flux_w_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux w interpolation"])

        # create rate operators evaluator
        self.rate_etor = black_oil_rate_evaluator(self.bo_bubble_pres_ev, self.bo_rs_ev, self.bo_xgo_ev,
                                                  self.bo_oil_dens_ev, self.bo_oil_visco_ev, self.bo_oil_sat_ev,
                                                  self.bo_oil_relperm_ev, self.bo_water_dens_ev, self.bo_water_sat_ev,
                                                  self.bo_water_visco_ev, self.bo_water_relperm_ev, self.bo_gas_dens_ev,
                                                  self.bo_gas_visco_ev, self.bo_gas_relperm_ev, self.bo_xcp_gas_ev,
                                                  self.bo_krow_ev, self.bo_krog_ev)

        try:
            self.rate_itor = rate_interpolator_name(self.rate_etor, index_vector([n_points] * self.n_components),
                                                    value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                    value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))
        except RuntimeError:
            self.rate_itor = rate_interpolator_name_long(self.rate_etor, index_vector([n_points] * self.n_components),
                                                         value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                         value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))

        # set up timers
        self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        self.rate_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"])

        # create engine according to physics selected
        self.engine = engine_name()

        # create well controls
        # gas stream
        # min_z is the minimum composition for interpolation
        # 2*min_z is the minimum composition for simulation
        # let`s take 3*min_z as the minimum composition for injection to be safely within both limits

        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                   self.n_components, rate,
                                                                   value_vector(inj_stream), self.rate_itor)

        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_components,
                                                                   self.n_components, rate,
                                                                   value_vector(inj_stream), self.rate_itor)

        self.new_rate_water_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 2, self.n_components,
                                                                     self.n_components,
                                                                     rate,
                                                                     value_vector(inj_stream), self.rate_itor)
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.phases, 2, self.n_components,
                                                                       self.n_components,
                                                                       rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: acc_flux_itor_name(new_acc_flux_etor,
                                                                              index_vector([n_points, n_points, n_points]),
                                                                              value_vector([min_p, min_z, min_z]),
                                                                              value_vector([max_p, 1 - min_z, 1 - min_z]))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        """""
        Function to set uniform initial reservoir condition
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]

    def set_nonuniform_initial_conditions(self, mesh, nonuniform_pressure, gas_comp, oil_comp):
        """""
        Function to set uniform initial reservoir condition
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = nonuniform_pressure

        # set initial composition
        z = np.array([gas_comp, oil_comp])
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = z[c]

    def set_uniform_saturation_initial_conditions(self, mesh, physics_filename, uniform_pressure,
                                                  uniform_saturation: list):
        """""
        Function to set uniform initial reservoir condition - saturation
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)

        state = value_vector([uniform_pressure, uniform_saturation[0], uniform_saturation[1]])
        Comp = saturation_composition()
        uniform_composition = Comp.evaluate(state, physics_filename)

        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]

    def set_nonuniform_saturation_initial_conditions(self, mesh, physics_filename, nonuniform_pressure,
                                                     gas_sat, oil_sat: list):
        """""
        Function to set nonuniform initial reservoir condition - saturation
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = np.array(nonuniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)

        Comp = saturation_composition()
        composition_ini = []

        for i in range(nb):
            state = value_vector([nonuniform_pressure[i], gas_sat[i], oil_sat[i]])
            composition_ini.append(Comp.evaluate(state, physics_filename))
        composition_ini = np.ravel(np.array(composition_ini))

        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = composition_ini[c::(self.n_components - 1)]