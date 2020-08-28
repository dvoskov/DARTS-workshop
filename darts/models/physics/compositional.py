from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from math import fabs

class Compositional:
    """"
       Class to generate compositional physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, components, n_points, min_p, max_p, min_z, with_gpu=0,  static_itor=False):
        """"
           Initialize Compositional class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - components: components name
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.components = components
        self.n_components = len(components)
        self.n_vars = self.n_components
        self.phases = ['gas', 'oil']
        self.vars = ['pressure'] + [c + ' composition' for c in components[:-1]]
        self.n_phases = len(self.phases)

        grav = 0
        try:
            scond = get_table_keyword(physics_filename, 'SCOND')[0]
            if len(scond) > 2 and fabs(scond[2]) < 1e-5:
                grav = 0
        except:
            grav = 1

        # Evaluate names of required classes depending on amount of components, self.phases, and selected physics
        # Different engines, accumulation_flux_operator_evaluators and number of operators for gravity on or off
        if with_gpu:
            plat = 'gpu'
            plat_itor = plat
        else:
            plat = 'cpu'
            plat_itor = 'i_d'
            if static_itor:
                plat_itor = 'static_i_d'

        if grav:
            engine_name = eval("engine_nc_cg_%s%d_%d" % (plat, self.n_vars, self.n_phases))
            acc_flux_etor_name = compositional_acc_flux_capillary_evaluator
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.n_phases
        else:
            engine_name = eval("engine_nc_%s%d" % (plat, self.n_vars))
            acc_flux_etor_name = compositional_acc_flux_evaluator
            self.n_ops = 2 * self.n_components


        acc_flux_itor_name = eval("operator_set_interpolator_%s_%d_%d" % (plat_itor, self.n_vars, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_%s_%d_%d" % (plat_itor, self.n_vars, self.n_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % ( self.n_vars, self.n_phases))

        sgof = get_table_keyword(physics_filename, 'SGOF')
        rock = get_table_keyword(physics_filename, 'ROCK')
        scond = get_table_keyword(physics_filename, 'SCOND')[0]
        temp = get_table_keyword(physics_filename, 'TEMP')[0][0]

        pres_sc = scond[0]
        temp_sc = scond[1]

        # create property evaluators
        self.gas_sat_ev = property_evaluator_iface()
        self.gas_relperm_ev = table_phase1_relative_permeability_evaluator(self.gas_sat_ev, sgof)
        self.oil_relperm_ev = table_phase2_relative_permeability_evaluator(self.gas_sat_ev, sgof)
        self.rock_compaction_ev = rock_compaction_evaluator(rock)
        self.pcgo_ev = table_phase_capillary_pressure_evaluator(self.gas_sat_ev, sgof)

        # create accumulation and flux operators evaluator
        if grav:
            self.acc_flux_etor = acc_flux_etor_name(self.n_components, self.n_phases, temp, self.components,
                                                self.oil_relperm_ev, self.gas_relperm_ev, self.pcgo_ev,
                                                self.rock_compaction_ev)
        else:
            self.acc_flux_etor = acc_flux_etor_name(self.n_components, self.n_phases, temp, self.components,
                                                    self.oil_relperm_ev, self.gas_relperm_ev, self.rock_compaction_ev)

        try:
            # try first to create interpolator with 4-byte index type
            self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points] * self.n_components),
                                                    value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                    value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))
        except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
            self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor,
                                                         index_vector([n_points] * self.n_components),
                                                         value_vector([min_p] + [min_z] * (self.n_components - 1)),
                                                         value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))

        # set up timers
        self.timer.node["jacobian assembly"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
        self.acc_flux_itor.init_timer_node(
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])

        self.rate_etor = compositional_rate_evaluator(self.n_components, self.n_phases, temp, pres_sc, temp_sc,
                                                      self.components,
                                                      self.oil_relperm_ev, self.gas_relperm_ev)
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

        # define well control factories

        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor,
                               index_vector([n_points] * self.n_components),
                               value_vector([min_p] + [min_z] * (self.n_components - 1)),
                               value_vector([max_p] + [1 - min_z] * (self.n_components - 1)))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
        """
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
