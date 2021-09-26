from darts.engines import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.do_properties_python import *

class property_deadoil_data():
    '''
    Class responsible for collecting all needed properties in dead oil (2p2c) simulation
    '''

    def __init__(self, physics_filename):
        pvdo = get_table_keyword(physics_filename, 'PVDO')
        swof = get_table_keyword(physics_filename, 'SWOF')
        pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
        dens = get_table_keyword(physics_filename, 'DENSITY')[0]
        rock = get_table_keyword(physics_filename, 'ROCK')

        self.surface_oil_dens = dens[0]
        self.surface_water_dens = dens[1]

        self.do_oil_dens_ev = dead_oil_table_density_evaluator(pvdo, self.surface_oil_dens)      # oil density
        self.do_wat_dens_ev = dead_oil_string_density_evaluator(pvtw, self.surface_water_dens)   # water density
        self.do_oil_visco_ev = dead_oil_table_viscosity_evaluator(pvdo)      # oil viscosity
        self.do_water_visco_ev = dead_oil_string_viscosity_evaluator(pvtw)   # water viscosity
        self.do_water_sat_ev = dead_oil_water_saturation_evaluator(pvdo,pvtw, self.surface_oil_dens, self.surface_water_dens)         # water saturation
        self.do_oil_relperm_ev = table_phase2_relative_permeability_evaluator(swof, pvdo,pvtw, self.surface_oil_dens, self.surface_water_dens)  # oil relperm
        self.do_wat_relperm_ev = table_phase1_relative_permeability_evaluator(swof, pvdo,pvtw, self.surface_oil_dens, self.surface_water_dens)  # wat relperm
        self.rock_compaction_ev = rock_compaction_evaluator(rock)        # rock compressibility
        self.do_pcow_ev = table_phase_capillary_pressure_evaluator(swof, pvdo,pvtw, self.surface_oil_dens, self.surface_water_dens)   # capillary pressure

class dead_oil_acc_flux_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        values[2] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[3] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0

class dead_oil_acc_flux_evaluator_well_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        values[2] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[3] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0

class dead_oil_acc_flux_capillary_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        pc = self.pc.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        # (1) water phase
        values[2] = wat_dens             # water density operator
        values[3] = 0                    # reference phase, pc = 0
        values[4] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[5] = 0                                   # water component in oil phase

        # (2) oil phase
        values[6] = oil_dens             # oil density operator
        values[7] = 0                    # pc, should be -pc, here we ignore pc
        values[8] = 0                                  # oil component in water phase
        values[9] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0


class dead_oil_acc_flux_capillary_evaluator_well_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        pc = self.pc.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        # (1) water phase
        values[2] = wat_dens             # water density operator
        values[3] = 0                    # reference phase, pc = 0
        values[4] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[5] = 0                                   # water component in oil phase

        # (2) oil phase
        values[6] = oil_dens             # oil density operator
        values[7] = 0                    # pc, should be -pc, here we ignore pc
        values[8] = 0                                  # oil component in water phase
        values[9] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0

class dead_oil_rate_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev
        self.surface_oil_dens = property_data.surface_oil_dens
        self.surface_wat_dens = property_data.surface_water_dens

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        # flux in reservoir condition
        wat_flux = wat_dens * (wat_relp / wat_visco)
        oil_flux = oil_dens * (oil_relp / oil_visco)

        # convert to surface condition
        values[0] = wat_flux / self.surface_wat_dens
        values[1] = oil_flux / self.surface_oil_dens
        values[2] = values[0] + values[1]

        return 0


class Saturation(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.oil_dens = property_data.do_oil_dens_ev
        self.wat_dens = property_data.do_wat_dens_ev
        self.oil_visco = property_data.do_oil_visco_ev
        self.wat_visco = property_data.do_water_visco_ev
        self.wat_sat = property_data.do_water_sat_ev
        self.oil_relp = property_data.do_oil_relperm_ev
        self.wat_relp = property_data.do_wat_relperm_ev
        self.pc = property_data.do_pcow_ev
        self.rock_cp = property_data.rock_compaction_ev

    def evaluate(self, state, values):
        oil_dens = self.oil_dens.evaluate(state)
        wat_dens = self.wat_dens.evaluate(state)
        oil_visco = self.oil_visco.evaluate(state)
        wat_visco = self.wat_visco.evaluate(state)
        wat_sat = self.wat_sat.evaluate(state)
        oil_relp = self.oil_relp.evaluate(state)
        wat_relp = self.wat_relp.evaluate(state)
        rock_cp = self.rock_cp.evaluate(state)

        values[0] = wat_sat


        return 0
