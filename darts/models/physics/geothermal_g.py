from darts.engines import *
from darts.physics import *
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.geothermal_operators_g import *

class Geothermal:
    """"
       Class to generate geothermal physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, n_points, min_p, max_p, min_e, max_e, mass_rate=False):
        """"
           Initialize Geothermal class.
           Arguments:
                - timer: time recording object
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_e, max_e: minimum and maximum enthalpy
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_e = min_e
        self.max_e = max_e
        self.n_components = 1
        self.thermal = 1
        self.n_vars = self.n_components + self.thermal * 1
        self.n_ops = 12
        if mass_rate:
            self.phases = ['water_mass', 'steam_mass', 'temperature', 'energy']
        else:
            self.phases = ['water', 'steam', 'temperature', 'energy']
        self.components = ['water']
        self.vars = ['pressure', 'enthalpy']
        self.n_phases = len(self.phases)
        self.n_rate_temp_ops = self.n_phases

        self.property_data = property_data()
        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        engine_name = eval("engine_nce_g_cpu%d_%d" % (self.n_components, len(self.phases)-2))
        
        acc_flux_etor_name = acc_flux_gravity_evaluator_python
        acc_flux_etor_name_well = acc_flux_gravity_evaluator_python_well
        acc_flux_itor_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_ops))
        acc_flux_itor_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_ops))

        if mass_rate:
            rate_etor_name = geothermal_mass_rate_custom_evaluator_python
        else:
            rate_etor_name = geothermal_rate_custom_evaluator_python

        rate_interpolator_name = eval("operator_set_interpolator_i_d_%d_%d" % (self.n_vars, self.n_rate_temp_ops))
        rate_interpolator_name_long = eval("operator_set_interpolator_l_d_%d_%d" % (self.n_vars, self.n_rate_temp_ops))
		
        self.acc_flux_etor = acc_flux_etor_name(self.property_data)
        self.acc_flux_etor_well = acc_flux_etor_name_well(self.property_data)

        try:
            # try first to create interpolator with 4-byte index type
            self.acc_flux_itor = acc_flux_itor_name(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))
            self.acc_flux_itor_well = acc_flux_itor_name(self.acc_flux_etor_well, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))
        except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
            self.acc_flux_itor = acc_flux_itor_name_long(self.acc_flux_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))
            self.acc_flux_itor_well = acc_flux_itor_name_long(self.acc_flux_etor_well, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))
                                                    

        # set up timers
        self.timer.node["jacobian assembly"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
        self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"] = timer_node()
        self.acc_flux_itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"])

        # create rate operators evaluator
        self.rate_etor = rate_etor_name(self.property_data)

        try:
            # try first to create interpolator with 4-byte index type
            self.rate_itor = rate_interpolator_name(self.rate_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))
        except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
            self.rate_itor = rate_interpolator_name_long(self.rate_etor, index_vector([n_points, n_points]),
                                                    value_vector([min_p, min_e]), value_vector([max_p, max_e]))


        # set up timers
        self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        self.rate_itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"])

        # create engine according to physics selected
        self.engine = engine_name()

        # create well controls
        # water stream
        # pure water injection at constant temperature

        self.water_inj_stream = value_vector([1.0])
        # water injection at constant temperature with bhp control
        self.new_bhp_water_inj = lambda bhp, temp: gt_bhp_temp_inj_well_control(self.phases, self.n_vars,
                                                                                bhp, temp,
                                                                                self.water_inj_stream, self.rate_itor)
        # water injection at constant temperature with volumetric rate control
        self.new_rate_water_inj = lambda rate, temp: gt_rate_temp_inj_well_control(self.phases, 0, self.n_vars,
                                                                                   rate, temp,
                                                                                   self.water_inj_stream, self.rate_itor)
        # water production with bhp control
        self.new_bhp_prod = lambda bhp: gt_bhp_prod_well_control(bhp)
        # water production with volumetric rate control
        self.new_rate_water_prod = lambda rate: gt_rate_prod_well_control(self.phases, 0, self.n_vars,
                                                                          rate, self.rate_itor)
        # water injection of constant enthalpy with mass rate control
        self.new_mass_rate_water_inj = lambda rate, enth: \
            gt_mass_rate_enthalpy_inj_well_control(self.phases, 0, self.n_vars,
                                                   self.water_inj_stream,
                                                   rate, enth,
                                                   self.rate_itor)
        # water production with mass rate control
        self.new_mass_rate_water_prod = lambda rate: gt_mass_rate_prod_well_control(self.phases, 0, self.n_vars,
                                                                                    rate, self.rate_itor)


    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components+1, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_temperature):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_temperature: uniform temperature setting
        """
        assert isinstance(mesh, conn_mesh)
        # nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        state = value_vector([uniform_pressure, 0])
        E = iapws_total_enthalpy_evalutor(uniform_temperature)
        enth = E.evaluate(state)

        enthalpy = np.array(mesh.enthalpy, copy=False)
        enthalpy.fill(enth)



