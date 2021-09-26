from darts.models.physics.geothermal_operators import *
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.physics_base import PhysicsBase
from darts.tools.keyword_file_tools import *


class Geothermal(PhysicsBase):
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

    def __init__(self, timer, n_points, min_p, max_p, min_e, max_e, mass_rate=False, grav=False,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        """"
           Initialize Geothermal class.
           Arguments:
                - timer: time recording object
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_e, max_e: minimum and maximum enthalpy
                - platform: target simulation platform - 'cpu' (default) or 'gpu'
                - itor_type: 'multilinear' (default) or 'linear' interpolator type
                - itor_mode: 'adaptive' (default) or 'static' OBL parametrization
                - itor_precision: 'd' (default) - double precision or 's' - single precision for interpolation
        """
        super().__init__(cache)
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_e = min_e
        self.max_e = max_e
        self.n_components = 1
        self.thermal = 1
        self.n_vars = self.n_components + self.thermal * 1
        self.n_ops = 2 * self.n_components + self.thermal * 6
        if mass_rate:
            self.phases = ['water_mass', 'steam_mass', 'temperature', 'energy']
        else:
            self.phases = ['water', 'steam', 'temperature', 'energy']
        self.components = ['water']
        self.vars = ['pressure', 'enthalpy']
        self.n_phases = len(self.phases)
        self.n_rate_temp_ops = self.n_phases

        self.n_axes_points = index_vector([n_points] * self.n_vars)
        self.n_axes_min = value_vector([min_p, min_e])
        self.n_axes_max = value_vector([max_p, max_e])

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        if grav:
            self.n_ops = 12
            self.property_data = property_data()
            self.engine = eval("engine_nce_g_%s%d_%d" % (platform, self.n_components, self.n_phases - 2))()
            self.acc_flux_etor = acc_flux_gravity_evaluator_python(self.property_data)
            self.acc_flux_etor_well = acc_flux_gravity_evaluator_python_well(self.property_data)
        else:
            self.n_ops = 2 * self.n_components + self.thermal * 6
            self.property_data = property_iapws_data()
            self.engine = eval("engine_nce_%s%d" % (platform, self.n_components))()
            self.acc_flux_etor = acc_flux_custom_iapws_evaluator_python(self.property_data)
            self.acc_flux_etor_well = acc_flux_custom_iapws_evaluator_python_well(self.property_data)

        self.acc_flux_itor = self.create_interpolator(self.acc_flux_etor, self.n_vars, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform, algorithm=itor_type, mode=itor_mode,
                                                      precision=itor_precision)

        self.acc_flux_itor_well = self.create_interpolator(self.acc_flux_etor_well, self.n_vars, self.n_ops,
                                                           self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                           platform=platform, algorithm=itor_type, mode=itor_mode,
                                                           precision=itor_precision)

        # create rate operators evaluator
        if mass_rate:
            self.rate_etor = geothermal_mass_rate_custom_evaluator_python(self.property_data)
        else:
            self.rate_etor = geothermal_rate_custom_evaluator_python(self.property_data)

        # interpolator platform is 'cpu' since rates are always computed on cpu
        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.n_rate_temp_ops,
                                                  self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                  platform='cpu', algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)

        # set up timers
        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.acc_flux_itor_well, 'well interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

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
                                                                                   self.water_inj_stream,
                                                                                   self.rate_itor)
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
            w.init_rate_parameters(self.n_components + 1, self.phases, self.rate_itor)

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

    def set_nonuniform_initial_conditions(self, mesh, pressure_grad, temperature_grad):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -pressure_grad, default unit [1/km]
            -temperature_grad, default unit [1/km]
        """
        assert isinstance(mesh, conn_mesh)

        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth / 1000 * pressure_grad + 1

        enthalpy = np.array(mesh.enthalpy, copy=False)
        temperature = depth / 1000 * temperature_grad + 293.15

        for j in range(mesh.n_blocks):
            state = value_vector([pressure[j], 0])
            E = iapws_total_enthalpy_evalutor(temperature[j])
            enthalpy[j] = E.evaluate(state)
