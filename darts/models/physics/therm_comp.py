from darts.engines import *
from darts.physics import *

from darts.models.physics.physics_base import PhysicsBase
# from darts.models.physics.iapws.custom_rock_property import *
from darts.tools.keyword_file_tools import *


class ThermComp(PhysicsBase):
    """"
       Class to generate thermal-compositional physics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """

    def __init__(self, timer, physics_filename, components, n_points, min_p, max_p, min_t, max_t, min_z, max_z=1,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        """"
           Initialize ThermComp class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - components: components name
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_t, max_t: minimum and maximum temperature
                - min_z, max_z: minimum and maximum composition
        """
        super().__init__(cache)
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_t = min_t
        self.max_t = max_t
        self.min_z = min_z
        if (max_z == 1):
            max_z = 1 - min_z
        self.max_z = max_z

        self.components = components
        self.n_components = len(components)
        self.n_vars = self.n_components + 1
        self.phases = ['gas', 'oil']
        self.vars = ['pressure'] + [c + ' composition' for c in components[:-1]] + ['temperature']
        self.n_phases = len(self.phases)
        self.thermal = 1
        self.n_axes_points = index_vector([n_points] * self.n_vars)
        self.n_axes_min = value_vector([min_p] + [min_z] * (self.n_components - 1) + [min_t])
        self.n_axes_max = value_vector([max_p] + [max_z] * (self.n_components - 1) + [max_t])

        # manual choice whether to use gravity or not
        grav = 0

        if grav:
            self.engine = eval("engine_nct_g_cpu%d_%d" % (self.n_vars, self.n_phases))()
            acc_flux_etor_name = therm_comp_acc_flux_grav_evaluator
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.thermal * 6
        else:
            self.engine = eval("engine_nct_cpu%d" % self.n_components)()
            acc_flux_etor_name = therm_comp_acc_flux_evaluator
            self.n_ops = 2 * self.n_components + self.thermal * 5

        sgof = get_table_keyword(physics_filename, 'SGOF')
        rock = get_table_keyword(physics_filename, 'ROCK')
        scond = get_table_keyword(physics_filename, 'SCOND')[0]

        # internal temperature for the class - not really needed in thermal case
        temp = get_table_keyword(physics_filename, 'TEMP')[0][0]

        pres_sc = scond[0]
        temp_sc = scond[1]

        # self.rock = [value_vector([1, 0, 273.15])]

        # create property evaluators
        self.gas_sat_ev = property_evaluator_iface()
        self.gas_relperm_ev = table_phase1_relative_permeability_evaluator(self.gas_sat_ev, sgof)
        self.oil_relperm_ev = table_phase2_relative_permeability_evaluator(self.gas_sat_ev, sgof)
        self.rock_compaction_ev = rock_compaction_evaluator(rock)
        self.rock_energy_ev = rock_energy_evaluator(rock)

        # create accumulation and flux operators evaluator
        self.acc_flux_etor = acc_flux_etor_name(self.n_components, self.n_phases, temp, self.components,
                                                self.oil_relperm_ev, self.gas_relperm_ev, self.rock_compaction_ev,
                                                self.rock_energy_ev)

        # create main interpolator for reservoir (platform should match engine platform)
        self.acc_flux_itor = self.create_interpolator(self.acc_flux_etor, self.n_vars, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform, algorithm=itor_type, mode=itor_mode,
                                                      precision=itor_precision)

        # create additional evaluator and interpolator for well rates
        self.rate_etor = therm_comp_rate_evaluator(self.n_components, self.n_phases, temp, pres_sc, temp_sc,
                                                   self.components,
                                                   self.oil_relperm_ev, self.gas_relperm_ev)

        # interpolator platform is 'cpu' since rates are always computed on cpu
        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.n_phases,
                                                  self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                  platform='cpu', algorithm=itor_type, mode=itor_mode,
                                                  precision=itor_precision)

        # set up timers
        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # define well control factories
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_vars,
                                                                               self.n_vars, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_vars,
                                                                     self.n_vars,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_vars,
                                                                     self.n_vars,
                                                                     rate, self.rate_itor)

        self.new_acc_flux_itor = lambda new_acc_flux_etor: \
            acc_flux_itor_name(new_acc_flux_etor,
                               index_vector([n_points] * (self.n_components + 1)),
                               value_vector([min_p] + [min_z] * (self.n_components - 1) + [min_t]),
                               value_vector([max_p] + [max_z] * (self.n_components - 1) + [max_t]))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition, uniform_temp: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
            -uniform_temp: uniform_temperature setting
        """
        assert isinstance(mesh, conn_mesh)
        assert len(uniform_composition) == self.n_components - 1

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        temp = np.array(mesh.temperature, copy=False)
        temp.fill(uniform_temp)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]
