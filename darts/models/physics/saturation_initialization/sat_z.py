from darts.physics import *
from darts.tools.keyword_file_tools import *

# initialize saturation
class saturation_composition(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state, physics_filename):
        # case 1: only one phase is present, unknown is P only
        if len(state) == 1:
            z = 1
            return z
        # case 2: two phases, unknown: P and z
        elif len(state) == 2:
            pvdo = get_table_keyword(physics_filename, 'PVDO')
            pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
            dens = get_table_keyword(physics_filename, 'DENSITY')[0]
            surface_oil_dens = dens[0]
            surface_water_dens = dens[1]
            # evaluate phase density under the initial pressure
            do_oil_dens_ev = dead_oil_table_density_evaluator(pvdo, surface_oil_dens)
            oil_dens = do_oil_dens_ev.evaluate(state)
            do_wat_dens_ev = dead_oil_string_density_evaluator(pvtw, surface_water_dens)
            water_dens = do_wat_dens_ev.evaluate(state)
            # saturation to corresponding composition
            z = water_dens * state[1] / (water_dens * state[1] + oil_dens * (1 - state[1]))
            return z
        # case 3: three phases, unknown: P, z1 and z2
        elif len(state) == 3:
            pvto = get_table_keyword(physics_filename, 'PVTO')
            pvdg = get_table_keyword(physics_filename, 'PVDG')
            pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
            dens = get_table_keyword(physics_filename, 'DENSITY')[0]
            pbub = get_table_keyword(physics_filename,'PBUB')[0]
            surface_oil_dens = dens[0]
            surface_water_dens = dens[1]
            surface_gas_dens = dens[2]
            pressure = state[0]
            # evaluate gas and water phase density under the initial pressure
            bo_gas_dens_ev = dead_oil_table_density_evaluator(pvdg, surface_gas_dens)
            gas_dens = bo_gas_dens_ev.evaluate(state)
            bo_water_dens_ev = dead_oil_string_density_evaluator(pvtw, surface_water_dens)
            water_dens = bo_water_dens_ev.evaluate(state)
            # evaluate oil density separately due to dissolved gas
            for i in range (len(pvto)):
                if pvto[i][0] == pvto[i+1][0]:
                    rs_index = i
                    break
            if pbub[0] < pvto[0][1]:
                rs_new = self.LinearExtrapolation(pvto, pbub[0], 1, 0)
                Bo_bub = self.LinearExtrapolation(pvto, pbub[0], 1, 2)
            elif pbub[0] > pvto[rs_index][1]:
                rs_new = self.SatExtrapolation(pvto, pbub[0], 1, 0, rs_index)
                Bo_bub = self.SatExtrapolation(pvto, pbub[0], 1, 2, rs_index)
            else:
                rs_new = self.LinearInterpolation(pvto, pbub[0], 1, 0)
                Bo_bub = self.LinearInterpolation(pvto, pbub[0], 1, 2)
            xgo = surface_gas_dens * rs_new / (surface_gas_dens * rs_new + surface_oil_dens)
            if pressure < pbub[0]:
                if pressure < pvto[0][1]:
                    Bo = self.LinearExtrapolation(pvto, pressure, 1, 2)
                else:
                    Bo = self.LinearInterpolation(pvto, pressure, 1, 2)
            else:
                pres_undersat = pressure + pvto[rs_index][1] - pbub[0]
                Bo_unsat = self.SatExtrapolation(pvto, pres_undersat, 1, 2, rs_index + 1)
                if pbub[0] < pvto[rs_index][1]:
                    Bo = Bo_unsat * Bo_bub / pvto[rs_index][2]
                else:
                    Bo = Bo_unsat - (pvto[rs_index][2] - Bo_bub)
            oil_dens = surface_oil_dens / Bo / (1 - xgo)

            # saturation to corresponding composition
            total_den = gas_dens * state[1] + oil_dens * state[2] + water_dens * (1 - state[1] - state[2])
            zw = water_dens * (1 - state[1] - state[2]) / total_den     # phase molar fraction[np]
            zo = oil_dens * (1 - xgo) * state[2] / total_den            # phase molar fraction[np]
            zg = 1 - zo - zw
            return zg, zo

    def LinearInterpolation(self, table, x_val, x_label, y_label):
        num = len(table)
        for i in range(num):
            if x_val - table[i][x_label] <= 0:
                y_val = table[i][y_label] + (x_val - table[i][x_label]) / (table[i][x_label] - table[i - 1][x_label]) \
                        * (table[i][y_label] - table[i - 1][y_label])
                break
        return y_val

    def LinearExtrapolation(self, table, x_val, x_label, y_label):
        num = len(table)
        if x_val < table[0][x_label]:
            y_val = table[0][y_label] + (x_val - table[0][x_label]) / (table[1][x_label] - table[0][x_label])\
                    *(table[1][y_label] - table[0][y_label])
        elif x_val > table[num-1][x_label]:
            y_val = table[num-1][y_label]+(x_val - table[num-1][x_label])/(table[num-1][x_label]-table[num-2][x_label])\
                    *(table[num-1][y_label] - table[num-2][y_label])
        return y_val

    def SatExtrapolation(self, table, x_val, x_label, y_label, lnum):
        y_val = table[lnum][y_label] + (x_val - table[lnum][x_label]) / (
                table[lnum][x_label] - table[lnum - 1][x_label]) * (table[lnum][y_label] - table[lnum - 1][y_label])
        return y_val