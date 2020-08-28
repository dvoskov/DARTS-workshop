from darts.physics import *
from darts.models.physics.iapws.iapws97 import _Region1, _Region2, _Region4, _Backward1_T_Ph, _Backward2_T_Ph, _Bound_Ph, _Bound_TP, _TSat_P, Pmin
from darts.models.physics.iapws._iapws import _D2O_Viscosity, _Viscosity
from scipy.optimize import newton


class water_density_property_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        #sat_steam_enthalpy = saturated_steam_enthalpy_evaluator()
        #sat_steam_enth = sat_steam_enthalpy.evaluate(state)
        #sat_water_enthalpy = saturated_water_enthalpy_evaluator()
        #sat_water_enth = sat_water_enthalpy.evaluate(state)
        #temperature = temperature_evaluator(sat_water_enthalpy, sat_steam_enthalpy)
        #temp = temperature.evaluate(state)
        temperature = temperature_region1_evaluator()
        temp = temperature.evaluate(state)
        water_density = 1 / _Region1(temp, float(state[0])*0.1)['v']
        return water_density / 18.015

class temperature_region1_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        return (_Backward1_T_Ph(float(state[0])*0.1, state[1] / 18.015))

class iapws_enthalpy_region1_evaluator(property_evaluator_iface):
    def __init__(self, temp):
        #super().__init__()
        self.temperature = temp
    def evaluate(self, state):
        return _Region1(self.temperature, float(state[0])*0.1)['h'] * 18.015        #kJ/kmol

class iapws_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = temperature_region1_evaluator()
        temp = temperature.evaluate(state)
        density = water_density_property_evaluator()
        den = density.evaluate(state) * 18.015
        return (_Viscosity(den, temp)*1000)



#====================================== Properties for Region 1 and 4 ============================================ 
class iapws_total_enthalpy_evalutor(property_evaluator_iface):
    def __init__(self, temp):
        self.T = temp
    def evaluate(self, state):
        P = state[0]*0.1
        region = _Bound_TP(self.T, P)
        if (region == 1):
            h = _Region1(self.T, P)["h"] * 18.015
        elif (region == 4):
            Steam_sat = iapws_steam_saturation_evaluator()
            rho_steam = iapws_steam_density_evaluator()
            rho_water = iapws_water_density_evaluator()
            x = Steam_sat.evaluate(state) * rho_steam.evaluate(state) / (Steam_sat.evaluate(state) * rho_steam.evaluate(state) + Steam_sat.evaluate(water) * rho_water.evaluate(state))
            h = _Region4(P, x)["h"] * 18.015
        elif (region == 2):
            h = _Region2(self.T, P)["h"] * 18.015
        else:
            raise NotImplementedError("Incoming out of bound")
        return h


class iapws_temperature_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            T = _Backward1_T_Ph(P, h)
        elif (region == 4):
            T = _TSat_P(P)
        elif (region == 2):
            T = _Backward2_T_Ph(P, h)
        else:
            raise NotImplementedError("Incoming out of bound")
        return T

class iapws_water_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin
        region = _Bound_Ph(P, h)
        if (region == 1):
            water_enth =  h
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               water_enth = _Region4(P, 0)["h"]
            else:
               raise NotImplementedError("Incoming out of bound")
        elif (region == 2):
            water_enth = 0
        else:
            print(region)
            raise NotImplementedError("Incoming out of bound")
        return water_enth * 18.015


class iapws_steam_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            steam_enth =  0
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               steam_enth = _Region4(P, 1)["h"]
            else:
               raise NotImplementedError("Incoming out of bound")
        elif (region == 2):
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_enth = _Region2(T, P)["h"]
        else:
            raise NotImplementedError("Incoming out of bound")
        return steam_enth * 18.015


class iapws_water_saturation_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            sw = 1
        elif (region == 4):
            hw = _Region4(P, 0)["h"]
            hs = _Region4(P, 1)["h"]
            rhow = 1 / _Region4(P, 0)["v"]
            rhos = 1 / _Region4(P, 1)["v"]
            sw = rhos * (hs - h) / (h * (rhow - rhos) - (hw * rhow - hs * rhos))
        elif (region == 2):
            sw = 0
        else:
             raise NotImplementedError("Incoming out of bound")
        return sw

class iapws_steam_saturation_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        water_saturation = iapws_water_saturation_evaluator()
        ss = 1 - water_saturation.evaluate(state)
        return ss

class iapws_water_relperm_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        water_saturation = iapws_water_saturation_evaluator()
        water_rp = water_saturation.evaluate(state)**1
        return water_rp

class iapws_steam_relperm_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        steam_saturation = iapws_steam_saturation_evaluator()
        steam_rp = steam_saturation.evaluate(state)**1
        return steam_rp


class iapws_water_density_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
            water_density = 1 / _Region1(T, P)['v']
        elif (region == 4):
            T = _TSat_P(P)
            if (T <= 623.15):
               water_density = 1 / _Region4(P, 0)['v']
            else:
               raise NotImplementedError("Incoming out of bound")
        elif (region == 2):
            water_density = 0
        else:
               raise NotImplementedError("Incoming out of bound")
        return water_density / 18.015


class iapws_steam_density_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            steam_density = 0
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               steam_density = 1 / _Region4(P, 1)['v']
            else:
               raise NotImplementedError("Incoming out of bound")
        elif (region == 2):
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_density = 1 / _Region2(T, P)["v"]
        else:
               raise NotImplementedError("Incoming out of bound")
        return steam_density / 18.015


class iapws_water_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = iapws_temperature_evaluator()
        T = temperature.evaluate(state)
        density = iapws_water_density_evaluator()
        den = density.evaluate(state) * 18.015
        return (_Viscosity(den, T)*1000)


class iapws_steam_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = iapws_temperature_evaluator()
        T = temperature.evaluate(state)
        density = iapws_steam_density_evaluator()
        den = density.evaluate(state) * 18.015
        return (_Viscosity(den, T)*1000)

