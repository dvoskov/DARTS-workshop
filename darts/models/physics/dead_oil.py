from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *


class DeadOil:
    """"
       Class to generate deadoil physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, n_points, min_p, max_p, min_z, isMP = False, with_gpu = False, static_itor=False):
        """"
           Initialize DeadOil class.
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
        self.n_components = 2
        self.n_vars = self.n_components
        self.phases = ['water', 'oil']
        self.components = ['water', 'oil']
        self.rate_phases = ['water', 'oil', 'liquid']
        self.vars = ['pressure', 'water composition']
        self.n_phases = len(self.phases)

        grav = 1
        try:
            scond = get_table_keyword(physics_filename, 'SCOND')[0]
            if len(scond) > 2 and fabs(scond[2]) < 1e-5:
                grav = 0
        except:
            grav = 1

        if with_gpu:
            plat = 'gpu'
            plat_itor = plat
        else:
            plat = 'cpu'
            plat_itor = 'i_d'
            if static_itor:
                plat_itor = 'static_i_d'

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        if grav:
            engine_name = eval("engine_nc_cg_%s%d_%d" % (plat, self.n_components, self.n_phases))
            acc_flux_etor_name = dead_oil_acc_flux_capillary_evaluator
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.n_phases
        else:
            if isMP:
                engine_name = eval("engine_nc_mp_%s%d" % (plat, self.n_components))
            else:
                engine_name = eval("engine_nc_%s%d" % (plat, self.n_components))
            acc_flux_etor_name = dead_oil_acc_flux_evaluator
            self.n_ops = 2 * self.n_components

        acc_flux_itor_name = eval("operator_set_interpolator_%s_%d_%d" % (plat_itor, self.n_vars, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_phases + 1))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_phases + 1))

        # read keywords from physics file
        pvdo = get_table_keyword(physics_filename, 'PVDO')
        swof = get_table_keyword(physics_filename, 'SWOF')
        pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
        dens = get_table_keyword(physics_filename, 'DENSITY')[0]
        rock = get_table_keyword(physics_filename, 'ROCK')

        swof_well = []
        swof_well.append(value_vector([swof[0][0],  swof[0][1],  swof[0][2],  0.0]))
        swof_well.append(value_vector([swof[-1][0], swof[-1][1], swof[-1][2], 0.0]))

        surface_oil_dens = dens[0]
        surface_water_dens = dens[1]

        # create property evaluators
        self.do_oil_dens_ev = dead_oil_table_density_evaluator(pvdo, surface_oil_dens)
        self.do_wat_dens_ev = dead_oil_string_density_evaluator(pvtw, surface_water_dens)
        self.do_oil_visco_ev = dead_oil_table_viscosity_evaluator(pvdo)
        self.do_water_visco_ev = dead_oil_string_viscosity_evaluator(pvtw)
        self.do_water_sat_ev = dead_oil_water_saturation_evaluator(self.do_wat_dens_ev, self.do_oil_dens_ev)
        self.do_oil_oil_relperm_ev = table_phase2_relative_permeability_evaluator(self.do_water_sat_ev, swof)
        self.do_oil_wat_relperm_ev = table_phase1_relative_permeability_evaluator(self.do_water_sat_ev, swof)
        self.rock_compaction_evaluator = rock_compaction_evaluator(rock)
        self.do_pcow_ev = table_phase_capillary_pressure_evaluator(self.do_water_sat_ev, swof)


        # create accumulation and flux operators evaluator
        if grav:
            self.do_pcow_w_ev = table_phase_capillary_pressure_evaluator(self.do_water_sat_ev, swof_well)

            self.acc_flux_etor = acc_flux_etor_name(self.do_oil_dens_ev, self.do_oil_visco_ev,
                                                    self.do_oil_oil_relperm_ev, self.do_wat_dens_ev,
                                                    self.do_water_sat_ev, self.do_water_visco_ev,
                                                    self.do_oil_wat_relperm_ev, self.do_pcow_ev,
                                                    self.rock_compaction_evaluator)
            self.acc_flux_w_etor = acc_flux_etor_name(self.do_oil_dens_ev, self.do_oil_visco_ev,
                                                    self.do_oil_oil_relperm_ev, self.do_wat_dens_ev,
                                                    self.do_water_sat_ev, self.do_water_visco_ev,
                                                    self.do_oil_wat_relperm_ev, self.do_pcow_w_ev,
                                                    self.rock_compaction_evaluator)
            try:
                self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
                self.acc_flux_w_itor = acc_flux_itor_name(self.acc_flux_w_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
            except RuntimeError:
                self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                        value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
                self.acc_flux_w_itor = acc_flux_itor_name_long(self.acc_flux_w_etor, index_vector([n_points, n_points]),
                                                          value_vector([min_p, min_z]),
                                                          value_vector([max_p, 1 - min_z]))
            # set up timers
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
            self.acc_flux_itor.init_timer_node(
                self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux w interpolation"] = timer_node()
            self.acc_flux_w_itor.init_timer_node(
                self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux w interpolation"])
        else:
            self.acc_flux_etor = acc_flux_etor_name(self.do_oil_dens_ev, self.do_oil_visco_ev,
                                                    self.do_oil_oil_relperm_ev, self.do_wat_dens_ev,
                                                    self.do_water_sat_ev, self.do_water_visco_ev,
                                                    self.do_oil_wat_relperm_ev, self.rock_compaction_evaluator)
            try:
                # try first to create interpolator with 4-byte index type
                self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                        value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
            except RuntimeError:
                # on exception (assume too small integer range) create interpolator with long index type
                self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                             value_vector([min_p, min_z]),
                                                             value_vector([max_p, 1 - min_z]))

            # create accumulation and flux operators interpolator
            # with adaptive uniform parametrization (accuracy is defined by 'n_points')
            # of compositional space (range is defined by 'min_p', 'max_p', 'min_z')
            # set up timers
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
            self.acc_flux_itor.init_timer_node(
                self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])


        # create rate operators evaluator
        self.rate_etor = dead_oil_rate_evaluator(self.do_oil_dens_ev, self.do_oil_visco_ev, self.do_oil_oil_relperm_ev,
                                                 self.do_wat_dens_ev, self.do_water_sat_ev, self.do_water_visco_ev,
                                                 self.do_oil_wat_relperm_ev)

        try:
            # try first to create interpolator with 4-byte index type
            self.rate_itor = rate_interpolator_name(self.rate_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
        except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
            self.rate_itor = rate_interpolator_name_long(self.rate_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))

        # set up timers
        self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        self.rate_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"])

        # create engine according to physics selected
        self.engine = engine_name()

        # create well controls
        # water stream
        # min_z is the minimum composition for interpolation
        # 2*min_z is the minimum composition for simulation
        # let`s take 3*min_z as the minimum composition for injection to be safely within both limits

        self.water_inj_stream = value_vector([1 - 3 * min_z])
        self.new_bhp_water_inj = lambda bhp: bhp_inj_well_control(bhp, self.water_inj_stream)
        self.new_rate_water_inj = lambda rate: rate_inj_well_control(self.rate_phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate,
                                                                     self.water_inj_stream, self.rate_itor)
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.rate_phases, 0, self.n_components,
                                                                       self.n_components,
                                                                       rate, self.rate_itor)

        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.rate_phases, 1, self.n_components,
                                                                       self.n_components,
                                                                       rate, self.rate_itor)
        self.new_rate_liq_prod = lambda rate: rate_prod_well_control(self.rate_phases, 2, self.n_components,
                                                                       self.n_components,
                                                                       rate, self.rate_itor)															   
        self.new_acc_flux_itor = lambda new_acc_flux_etor: acc_flux_itor_name(new_acc_flux_etor,
                                                                              index_vector([n_points, n_points]),
                                                                              value_vector([min_p, min_z]),
                                                                              value_vector([max_p, 1 - min_z]))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.rate_phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
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

        # injectivity fix
        # composition[6569] = 1.0 - 1e-11
        # composition[19769] = 1.0 - 1e-11
