from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.do_operator_python import *


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
    def __init__(self, timer, physics_filename, n_points, min_p, max_p, min_z):
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
        self.vars = ['pressure', 'water composition']
        self.n_phases = len(self.phases)

        grav = 1
        try:
            scond = get_table_keyword(physics_filename, 'SCOND')[0]
            if len(scond) > 2 and fabs(scond[2]) < 1e-5:
                grav = 0
        except:
            grav = 1

        self.property_data = property_deadoil_data(physics_filename)

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        if grav:
            engine_name = eval("engine_nc_cg_cpu%d_%d" % (self.n_components, self.n_phases))
            acc_flux_etor_name = dead_oil_acc_flux_capillary_evaluator_python
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.n_phases
        else:
            engine_name = eval("engine_nc_cpu%d" % self.n_components)
            acc_flux_etor_name = dead_oil_acc_flux_evaluator_python
            self.n_ops = 2 * self.n_components


        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_phases))

        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_phases))

        # # create accumulation and flux operators evaluator
        self.acc_flux_etor = acc_flux_etor_name(self.property_data)

        if grav:

            try:
                self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))

            except RuntimeError:
                self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                        value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))

            # set up timers
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
            self.acc_flux_itor.init_timer_node(
                self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])
            self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux w interpolation"] = timer_node()

        else:
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


        self.rate_etor = dead_oil_rate_evaluator_python(self.property_data)


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
        self.new_bhp_water_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_water_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate,
                                                                     value_vector(inj_stream), self.rate_itor)
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                       self.n_components,
                                                                       rate, self.rate_itor)

        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
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
