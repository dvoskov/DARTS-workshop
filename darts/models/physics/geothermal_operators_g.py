from darts.engines import *
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
from darts.physics import *


class property_data():
    '''
    Class resposible for collecting all needed properties in geothermal simulation
    '''
    def __init__(self):
        self.rock = [value_vector([1, 0, 273.15])]
        # properties implemented in C++
        # self.sat_steam_enthalpy = saturated_steam_enthalpy_evaluator()
        # self.sat_water_enthalpy = saturated_water_enthalpy_evaluator()
        # self.water_enthalpy = water_enthalpy_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.steam_enthalpy = steam_enthalpy_evaluator(self.sat_steam_enthalpy, self.sat_water_enthalpy)
        # self.sat_water_density = saturated_water_density_evaluator(self.sat_water_enthalpy)
        # self.sat_steam_density = saturated_steam_density_evaluator(self.sat_steam_enthalpy)
        # self.water_density = water_density_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.steam_density = steam_density_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.temperature = temperature_evaluator(self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.water_saturation = water_saturation_evaluator(self.sat_water_density, self.sat_steam_density,
        #                                                    self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.steam_saturation = steam_saturation_evaluator(self.sat_water_density, self.sat_steam_density,
        #                                                    self.sat_water_enthalpy, self.sat_steam_enthalpy)
        # self.water_viscosity = water_viscosity_evaluator(self.temperature)
        # self.steam_viscosity = steam_viscosity_evaluator(self.temperature)
        # self.water_relperm = water_relperm_evaluator(self.water_saturation)
        # self.steam_relperm = steam_relperm_evaluator(self.steam_saturation)
        # self.rock_compaction = custom_rock_compaction_evaluator(self.rock)
        # self.rock_energy = custom_rock_energy_evaluator(self.rock)
        
        # properties implemented in python (the IAPWS package)
        self.temperature = iapws_temperature_evaluator()                       # Create temperature object
        self.water_enthalpy = iapws_water_enthalpy_evaluator()                 # Create water_enthalpy object
        self.steam_enthalpy = iapws_steam_enthalpy_evaluator()                 # Create steam_enthalpy object
        self.water_saturation = iapws_water_saturation_evaluator()             # Create water_saturation object
        self.steam_saturation = iapws_steam_saturation_evaluator()             # Create steam_saturation object
        self.water_relperm = iapws_water_relperm_evaluator()                   # Create water_relperm object
        self.steam_relperm = iapws_steam_relperm_evaluator()                   # Create steam_relperm object
        self.water_density = iapws_water_density_evaluator()                   # Create water_density object
        self.steam_density = iapws_steam_density_evaluator()                   # Create steam_density object
        self.water_viscosity = iapws_water_viscosity_evaluator()               # Create water_viscosity object
        self.steam_viscosity = iapws_steam_viscosity_evaluator()               # Create steam_viscosity object
        self.rock_compaction = custom_rock_compaction_evaluator(self.rock)     # Create rock_compaction object
        self.rock_energy = custom_rock_energy_evaluator(self.rock)             # Create rock_energy object


class acc_flux_gravity_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.temperature        = property_data.temperature
        self.water_enthalpy     = property_data.water_enthalpy
        self.steam_enthalpy     = property_data.steam_enthalpy
        self.water_saturation   = property_data.water_saturation
        self.steam_saturation   = property_data.steam_saturation
        self.water_relperm      = property_data.water_relperm
        self.steam_relperm      = property_data.steam_relperm
        self.water_density      = property_data.water_density
        self.steam_density      = property_data.steam_density
        self.water_viscosity    = property_data.water_viscosity
        self.steam_viscosity    = property_data.steam_viscosity
        self.rock_compaction    = property_data.rock_compaction
        self.rock_energy        = property_data.rock_energy

    def evaluate(self, state, values):

        water_enth = self.water_enthalpy.evaluate(state)
        steam_enth = self.steam_enthalpy.evaluate(state)
        water_den  = self.water_density.evaluate(state)
        steam_den  = self.steam_density.evaluate(state)
        water_sat  = self.water_saturation.evaluate(state)
        steam_sat  = self.steam_saturation.evaluate(state)
        temp       = self.temperature.evaluate(state)
        water_rp   = self.water_relperm.evaluate(state)
        steam_rp   = self.steam_relperm.evaluate(state)
        water_vis  = self.water_viscosity.evaluate(state)
        steam_vis  = self.steam_viscosity.evaluate(state)
        pore_volume_factor = self.rock_compaction.evaluate(state)
        rock_int_energy    = self.rock_energy.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis
        values[2] = steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                           - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = water_enth * water_den * water_rp / water_vis
        values[6] = steam_enth * steam_den * steam_rp / steam_vis
        # fluid conduction
        values[7] = 2.0*86.4
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = temp
        # water density
        values[10] = water_den
        # steam density
        values[11] = steam_den

        return 0

class acc_flux_gravity_evaluator_python_well(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.temperature        = property_data.temperature
        self.water_enthalpy     = property_data.water_enthalpy
        self.steam_enthalpy     = property_data.steam_enthalpy
        self.water_saturation   = property_data.water_saturation
        self.steam_saturation   = property_data.steam_saturation
        self.water_relperm      = property_data.water_relperm
        self.steam_relperm      = property_data.steam_relperm
        self.water_density      = property_data.water_density
        self.steam_density      = property_data.steam_density
        self.water_viscosity    = property_data.water_viscosity
        self.steam_viscosity    = property_data.steam_viscosity
        self.rock_compaction    = property_data.rock_compaction
        self.rock_energy        = property_data.rock_energy

    def evaluate(self, state, values):

        water_enth = self.water_enthalpy.evaluate(state)
        steam_enth = self.steam_enthalpy.evaluate(state)
        water_den  = self.water_density.evaluate(state)
        steam_den  = self.steam_density.evaluate(state)
        water_sat  = self.water_saturation.evaluate(state)
        steam_sat  = self.steam_saturation.evaluate(state)
        temp       = self.temperature.evaluate(state)
        water_rp   = self.water_relperm.evaluate(state)
        steam_rp   = self.steam_relperm.evaluate(state)
        water_vis  = self.water_viscosity.evaluate(state)
        steam_vis  = self.steam_viscosity.evaluate(state)
        pore_volume_factor = self.rock_compaction.evaluate(state)
        rock_int_energy    = self.rock_energy.evaluate(state)
        pressure = state[0]

        # mass accumulation
        values[0] = pore_volume_factor * (water_den * water_sat + steam_den * steam_sat)
        # mass flux
        values[1] = water_den * water_rp / water_vis
        values[2] = steam_den * steam_rp / steam_vis
        # fluid internal energy = water_enthalpy + steam_enthalpy - work
        # (in the following expression, 100 denotes the conversion factor from bars to kJ/m3)
        values[3] = pore_volume_factor * (water_den * water_sat * water_enth + steam_den * steam_sat * steam_enth
                                           - 100 * pressure)
        # rock internal energy
        values[4] = rock_int_energy / pore_volume_factor
        # energy flux
        values[5] = water_enth * water_den * water_rp / water_vis
        values[6] = steam_enth * steam_den * steam_rp / steam_vis
        # fluid conduction
        values[7] = 0.0 * 86.4
        # rock conduction
        values[8] = 1 / pore_volume_factor
        # temperature
        values[9] = temp
        # water density
        values[10] = water_den
        # steam density
        values[11] = steam_den

        return 0
    
class geothermal_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.temperature        = property_data.temperature
        self.water_enthalpy     = property_data.water_enthalpy
        self.steam_enthalpy     = property_data.steam_enthalpy
        self.water_saturation   = property_data.water_saturation
        self.steam_saturation   = property_data.steam_saturation
        self.water_relperm      = property_data.water_relperm
        self.steam_relperm      = property_data.steam_relperm
        self.water_density      = property_data.water_density
        self.steam_density      = property_data.steam_density
        self.water_viscosity    = property_data.water_viscosity
        self.steam_viscosity    = property_data.steam_viscosity

    def evaluate(self, state, values):
        water_den = self.water_density.evaluate(state)
        steam_den = self.steam_density.evaluate(state)
        water_sat = self.water_saturation.evaluate(state)
        steam_sat = self.steam_saturation.evaluate(state)
        water_rp  = self.water_relperm.evaluate(state)
        steam_rp  = self.steam_relperm.evaluate(state)
        water_vis = self.water_viscosity.evaluate(state)
        steam_vis = self.steam_viscosity.evaluate(state)
        water_enth = self.water_enthalpy.evaluate(state)
        steam_enth = self.steam_enthalpy.evaluate(state)
        temp = self.temperature.evaluate(state)

        total_density = water_sat * water_den + steam_sat * steam_den

        # water volumetric rate
        values[0] = water_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # steam volumetric rate
        values[1] = steam_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # temperature
        values[2] = temp
        # energy rate
        values[3] = water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis

        return 0
		
class geothermal_mass_rate_custom_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.water_density      = property_data.water_density
        self.steam_density      = property_data.steam_density
        self.water_saturation   = property_data.water_saturation
        self.steam_saturation   = property_data.steam_saturation
        self.water_relperm      = property_data.water_relperm
        self.steam_relperm      = property_data.steam_relperm		
        self.water_viscosity    = property_data.water_viscosity
        self.steam_viscosity    = property_data.steam_viscosity
        self.temperature        = property_data.temperature
        self.water_enth         = property_data.water_enthalpy
        self.steam_enth         = property_data.steam_enthalpy

		
    def evaluate(self, state, values):
        water_den = self.water_density.evaluate(state)
        steam_den = self.steam_density.evaluate(state)
        water_sat = self.water_saturation.evaluate(state)
        steam_sat = self.steam_saturation.evaluate(state)
        water_rp  = self.water_relperm.evaluate(state)
        steam_rp  = self.steam_relperm.evaluate(state)
        water_vis = self.water_viscosity.evaluate(state)
        steam_vis = self.steam_viscosity.evaluate(state)
        temp      = self.temperature.evaluate(state)
        water_enth= self.water_enth.evaluate(state)
        steam_enth= self.steam_enth.evaluate(state)

        total_density = water_sat * water_den + steam_sat * steam_den

        # water mass rate
        values[0] = water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis
        # steam mass rate
        values[1] = steam_sat * (water_den * water_rp / water_vis + steam_den * steam_rp / steam_vis) / total_density
        # temperature
        values[2] = temp
        # energy rate
        values[3] = water_enth * water_den * water_rp / water_vis + steam_enth * steam_den * steam_rp / steam_vis
        
        return 0