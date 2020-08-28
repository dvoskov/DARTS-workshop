from darts.physics import *
from darts.models.physics.iapws.iapws97 import _Backward1_T_Ph
from darts.models.physics.iapws.iapws_property import iapws_temperature_evaluator

class custom_rock_energy_evaluator(property_evaluator_iface):
     def __init__(self, rock):
        super().__init__()
        self.rock_table = rock
     def evaluate(self, state):
        sat_steam_enthalpy = saturated_steam_enthalpy_evaluator()
        sat_water_enthalpy = saturated_water_enthalpy_evaluator()
        T = temperature_evaluator(sat_water_enthalpy, sat_steam_enthalpy)
        # T = iapws_temperature_evaluator()
        temperature = T.evaluate(state)
        temperature_ref = self.rock_table[0][2]
        heat_constant = 1
        return (heat_constant * (temperature - temperature_ref))

class custom_rock_compaction_evaluator(property_evaluator_iface):
    def __init__(self, rock):
        super().__init__()
        self.rock_table = rock
    def evaluate(self, state):
        pressure = state[0]
        pressure_ref = self.rock_table[0][0]
        compressibility = self.rock_table[0][1]
        return (1.0 + compressibility * (pressure - pressure_ref))
