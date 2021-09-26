from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.do_operator_python import *
from darts.models.physics.physics_base import PhysicsBase


class DeadOil(PhysicsBase):
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
    def __init__(self, timer, physics_filename, n_points, min_p, max_p, min_z, platform='cpu', itor_type='multilinear',
                 itor_mode='adaptive', itor_precision='d', cache=True):
        """"
           Initialize DeadOil class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
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
        self.min_z = min_z
        self.n_components = 2
        self.n_vars = self.n_components
        self.phases = ['water', 'oil']
        self.components = ['water', 'oil']
        self.rate_phases = ['water', 'oil', 'liquid']
        self.vars = ['pressure', 'water composition']
        self.n_phases = len(self.phases)
        self.n_axes_points = index_vector([n_points] * self.n_vars)
        self.n_axes_min = value_vector([min_p] + [min_z] * (self.n_components - 1))
        self.n_axes_max = value_vector([max_p] + [1 - min_z] * (self.n_components - 1))

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
            self.acc_flux_etor = dead_oil_acc_flux_capillary_evaluator_python(self.property_data)
            self.acc_flux_w_etor = dead_oil_acc_flux_capillary_evaluator_well_python(self.property_data)
            self.property_etor = Saturation(self.property_data)
            self.n_ops = self.n_components + self.n_components * self.n_phases + self.n_phases + self.n_phases
        else:
            engine_name = eval("engine_nc_cpu%d" % self.n_components)
            self.acc_flux_etor = dead_oil_acc_flux_evaluator_python(self.property_data)
            self.acc_flux_w_etor = dead_oil_acc_flux_evaluator_well_python(self.property_data)
            self.property_etor = Saturation(self.property_data)
            self.n_ops = 2 * self.n_components

        # create main interpolator for reservoir (platform should match engine platform)
        self.acc_flux_itor = self.create_interpolator(self.acc_flux_etor, self.n_vars, self.n_ops, self.n_axes_points,
                                                      self.n_axes_min, self.n_axes_max, platform=platform)

        self.acc_flux_w_itor = self.create_interpolator(self.acc_flux_w_etor, self.n_vars, self.n_ops, self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)

        self.rate_etor = dead_oil_rate_evaluator_python(self.property_data)
        # interpolator platform is 'cpu' since rates are always computed on cpu
        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.n_phases + 1, self.n_axes_points,
                                                  self.n_axes_min, self.n_axes_max, platform='cpu', algorithm=itor_type,
                                                  mode=itor_mode,
                                                  precision=itor_precision)

        self.property_itor = self.create_interpolator(self.property_etor, self.n_vars, self.n_ops, self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)

        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.acc_flux_w_itor, 'well interpolation')
        self.create_itor_timers(self.property_itor, 'property interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # create engine according to physics selected
        self.engine = engine_name()

        # create well controls
        # water stream
        self.new_bhp_water_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_water_inj = lambda rate, inj_stream: rate_inj_well_control(self.rate_phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate,
                                                                     value_vector(inj_stream), self.rate_itor)
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
