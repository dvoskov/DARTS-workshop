import numpy as np
from darts.physics import *
from utils.interpolation import TableInterpolation
from darts.tools.keyword_file_tools import *

class DensityGas:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][2]
        self.table = get_table_keyword(self.pvt, 'PVDG')

    def evaluate(self, pres, pbub, xgo):
        pres_index = 0    # first column in the table - pressure
        Bg_index = 1      # second column in the table - volume factor
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            Bg = Table.LinearExtraP(self.table, pres, pres_index, Bg_index)
        else:
            Bg = Table.LinearInterP(self.table, pres, pres_index, Bg_index)

        return self.dens_sc / Bg


class DensityWat:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][1]
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, pres, pbub, xgo):
        X = self.table[2] * (pres - self.table[0])
        Bw = self.table[1] / (1 + X + X * X / 2)

        return self.dens_sc / Bw


class ViscGas:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.table = get_table_keyword(self.pvt, 'PVDG')

    def evaluate(self, pres, pbub):
        pres_index = 0  # first column in the table - pressure
        vgas_index = 2   # third column in the table - viscosity
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            visco_gas = Table.LinearExtraP(self.table, pres, pres_index, vgas_index)
        else:
            visco_gas = Table.LinearInterP(self.table, pres, pres_index, vgas_index)

        return visco_gas


class ViscWat:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, pres, pbub):
        Y = -self.table[4] * (pres - self.table[0])
        visco_wat = self.table[3] / (1 + Y + Y * Y / 2)

        return visco_wat




class flash_black_oil:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.oil_dens = self.dens[0]
        self.gas_dens = self.dens[2]
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)


    def bubble_point_pressure(self, pres, z):
        zg = z[0]
        zo = z[1]
        sat_rs = self.oil_dens / (self.gas_dens * (zo + zg) / zg - self.gas_dens)
        rs_index = 0        # first column in the table - rs
        pres_index = 1      # second column in the table - pressure
        Table = TableInterpolation()

        # find the index of saturated Rs
        for i in range(self.len_table - 1):
            if (self.table[i][0] == self.table[i+1][0]):
                num = i
                break

        if (sat_rs < self.table[0][0]):
            pbub = Table.LinearExtraP(self.table, sat_rs, rs_index, pres_index)
        elif (sat_rs > self.table[self.len_table-1][0]):
            pbub = Table.SatExtrapolation(self.table, sat_rs, rs_index, pres_index, num)
        else:
            pbub = Table.LinearInterP(self.table, sat_rs, rs_index, pres_index)

        if pres>pbub:
            pbub = pbub
        else:
            pbub = pres

        return pbub

    def evaluate(self, pres, z):
        zg = z[0]
        zo = z[1]
        zw = 1 - zg - zo

        pbub = self.bubble_point_pressure(pres, z)

        rs_index = 0  # first column in the table - rs
        pres_index = 1    # second column in the table - pressure
        Table = TableInterpolation()

        # find the index of saturated Rs
        for i in range(self.len_table - 1):
            if (self.table[i][0] == self.table[i + 1][0]):
                num = i
                break

        # undersaturated condition
        if (pres > pbub):
            rs = self.oil_dens / (self.gas_dens * (zo+zg) / zg - self.gas_dens)
        # saturated condition
        else:
            if pres<self.table[0][1]:
                rs = Table.LinearExtraP(self.table, pres, pres_index, rs_index)
            elif pres>self.table[num][1]:
                rs = Table.SatExtrapolation(self.table, pres, pres_index, rs_index, num)
            else:
                rs = Table.LinearInterP(self.table, pres, pres_index, rs_index)

        xgo = self.gas_dens * rs / (self.oil_dens + self.gas_dens * rs)

        V = 1 - zw - zo / (1 - xgo)

        return xgo, V, pbub


class DensityOil:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][0]
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)

    def evaluate(self, pres, pbub, xgo):
        pres_index = 1
        Bo_index = 2

        Table = TableInterpolation()

        # find the index of max Bo
        for i in range(self.len_table - 1):
            if (self.table[i][2] >= self.table[i + 1][2]):
                num = i
                break

        # calculate the saturated Bo
        if (pbub < self.table[0][1]):
            Bo_bub = Table.LinearExtraP(self.table, pbub, pres_index, Bo_index)
        elif (pbub > self.table[num][1]):
            Bo_bub = Table.SatExtrapolation(self.table, pbub, pres_index, Bo_index, num)
        else:
            Bo_bub = Table.LinearInterP(self.table, pbub, pres_index, Bo_index)

        # calculate Bo in current pressure
        # (1) saturated condition
        if pres < pbub:
            if (pres < self.table[0][1]):
                Bo = Table.LinearExtraP(self.table, pres, pres_index, Bo_index)
            elif (pres > self.table[num][1]):
                Bo = Table.SatExtrapolation(self.table, pres, pres_index, Bo_index, num)
            else:
                Bo = Table.LinearInterP(self.table, pres, pres_index, Bo_index)
        # (2) undersaturated condition
        else:
            pres_undersat = pres + self.table[num][1] - pbub
            Bo_undersat = Table.SatExtrapolation(self.table, pres_undersat, pres_index, Bo_index, num+1)
            if (pbub < self.table[num][1]):
                Bo = Bo_undersat * Bo_bub / self.table[num][2]
            else:
                Bo = Bo_undersat - (self.table[num][2] - Bo_bub)

        return self.dens_sc / Bo / (1 - xgo)


class ViscOil:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.table = get_table_keyword(self.pvt, 'PVTO')
        self.len_table = len(self.table)

    def evaluate(self, pres, pbub):
        pres_index = 1
        visco_index = 3

        Table = TableInterpolation()

        # find the index of min visco
        for i in range(self.len_table - 1):
            if (self.table[i][3] <= self.table[i + 1][3]):
                num = i
                break

        # calculate the saturated viscosity
        if (pbub < self.table[0][1]):
            visco_bub = Table.LinearExtraP(self.table, pbub, pres_index, visco_index)
        elif (pbub > self.table[num][1]):
            visco_bub = Table.SatExtrapolation(self.table, pbub, pres_index, visco_index, num)
        else:
            visco_bub = Table.LinearInterP(self.table, pbub, pres_index, visco_index)

        # calculate viscosity at current pressure
        # (1) saturated condition
        if pres < pbub:
            if (pres < self.table[0][1]):
                visco = Table.LinearExtraP(self.table, pres, pres_index, visco_index)
            elif (pres > self.table[num][1]):
                visco = Table.SatExtrapolation(self.table, pres, pres_index, visco_index, num)
            else:
                visco = Table.LinearInterP(self.table, pres, pres_index, visco_index)
        # (2) undersaturated condition
        else:
            pres_undersat = pres + self.table[num][1] - pbub
            visco_undersat = Table.SatExtrapolation(self.table, pres_undersat, pres_index, visco_index, num + 1)
            if (pbub < self.table[num][1]):
                visco = visco_undersat * visco_bub / self.table[num][3]
            else:
                visco = visco_undersat - (self.table[num][3] - visco_bub)

        return visco


class WatRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, gas_sat, wat_sat):
        wat_index = 0
        krw_index = 1

        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            krw = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, krw_index)
        else:
            krw = Table.LinearInterP(self.SWOF, wat_sat, wat_index, krw_index)

        return krw


class GasRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')

    def evaluate(self, gas_sat, wat_sat):
        gas_index = 0
        krg_index = 1

        Table = TableInterpolation()
        if (gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]):
            krg = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, krg_index)
        else:
            krg = Table.LinearInterP(self.SGOF, gas_sat, gas_index, krg_index)

        return krg


# here we use Stone I model to calculate oil relperm
class OilRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def Krog(self, gas_sat):
        gas_index = 0
        krog_index = 2

        Table = TableInterpolation()
        if (gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]):
            krog = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, krog_index)
        else:
            krog = Table.LinearInterP(self.SGOF, gas_sat, gas_index, krog_index)

        return krog

    def Krow(self, wat_sat):
        wat_index = 0
        krow_index = 2

        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            krow = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, krow_index)
        else:
            krow = Table.LinearInterP(self.SWOF, wat_sat, wat_index, krow_index)

        return krow


    def evaluate(self, gas_sat, wat_sat):
        len_SWOF = len(self.SWOF)
        len_SGOF = len(self.SGOF)
        Sorw = 1 - self.SWOF[len_SWOF - 1][0]
        Sorg = 1 - self.SGOF[len_SGOF - 1][0]
        Swc = self.SWOF[0][0]
        MINIMAL_FOR_COMPARE = 1e-12
        Krocw = self.SWOF[0][2]
        SatLimit = 1 - 2 * MINIMAL_FOR_COMPARE
        oil_sat = 1 - wat_sat - gas_sat

        krow = self.Krow(wat_sat)
        krog = self.Krog(gas_sat)

        if (gas_sat < MINIMAL_FOR_COMPARE):
            kro = krow        # water-oil two phases
        elif (wat_sat < Swc):
            kro = krog        # water phase not mobile
        else:
            # Stone I model -> alpha
            alpha = 1 - gas_sat / (1 - min(Swc + Sorg, SatLimit))
            # -> Som
            Som = alpha * Sorw + (1.0 - alpha) * Sorg
            # -> denom
            denom = 1.0 / (1.0 - min(Swc + Som, SatLimit))
            # Normalized saturations
            if (oil_sat - Som) > 0:
                Ma = oil_sat - Som
            else:
                Ma = 0
            SoStar = Ma * denom
            SwStar = min(wat_sat - Swc, SatLimit) * denom
            SgStar = min(gas_sat, SatLimit) * denom

            kro = (SoStar * krow * krog) / (Krocw * (1.0 - SwStar) * (1.0 - SgStar))

        return kro


class RockCompactionEvaluator:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.rock_table = get_table_keyword(self.pvt, 'ROCK')

    def evaluate(self, state):
        pressure = state[0]
        pressure_ref = self.rock_table[0][0]
        compressibility = self.rock_table[0][1]

        return 1.0 + compressibility * (pressure - pressure_ref)


# capillary pressure based on table
class CapillaryPressurePcow(property_evaluator_iface):
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, wat_sat):
        wat_index = 0
        pc_index = 3

        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            pcow = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, pc_index)
        else:
            pcow = Table.LinearInterP(self.SWOF, wat_sat, wat_index, pc_index)

        return pcow

class CapillaryPressurePcgo(property_evaluator_iface):
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SGOF = get_table_keyword(self.pvt, 'SGOF')

    def evaluate(self, gas_sat):
        gas_index = 0
        pc_index = 3

        Table = TableInterpolation()
        if (gas_sat < self.SGOF[0][0] or gas_sat > self.SGOF[len(self.SGOF) - 1][0]):
            pcgo = Table.SCALExtraP(self.SGOF, gas_sat, gas_index, pc_index)
        else:
            pcgo = Table.LinearInterP(self.SGOF, gas_sat, gas_index, pc_index)

        return pcgo