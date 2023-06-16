import numpy as np
from darts.tools.interpolation import TableInterpolation
from darts.tools.keyword_file_tools import get_table_keyword
from darts.engines import property_evaluator_iface

class DensityOil:
     def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][0]
        self.table = get_table_keyword(self.pvt, 'PVDO')

     def evaluate(self, state):
        pres = state[0]
        pres_index = 0             # first column in the table - pressure
        Bo_index   = 1             # second column in the table - volume factor
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            Bo = Table.LinearExtraP(self.table, pres, pres_index, Bo_index)
        else:
            Bo = Table.LinearInterP(self.table, pres, pres_index, Bo_index)

        return self.dens_sc / Bo


class DensityWat:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.dens_sc = get_table_keyword(self.pvt, 'DENSITY')[0][1]
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, state):
        pres = state[0]
        X = self.table[2] * (pres - self.table[0])
        Bw = self.table[1] / (1 + X + X * X / 2)

        return self.dens_sc / Bw


class ViscoOil(property_evaluator_iface):
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.table = get_table_keyword(self.pvt, 'PVDO')

    def evaluate(self, state):
        pres = state[0]
        pres_index = 0  # first column in the table - pressure
        muo_index = 2   # third column in the table - viscosity
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            visco_oil = Table.LinearExtraP(self.table, pres, pres_index, muo_index)
        else:
            visco_oil = Table.LinearInterP(self.table, pres, pres_index, muo_index)

        return visco_oil

class ViscoWat:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.table = get_table_keyword(self.pvt, 'PVTW')[0]

    def evaluate(self, state):
        pres = state[0]
        Y = -self.table[4] * (pres - self.table[0])
        visco_wat = self.table[3] / (1 + Y + Y * Y / 2)

        return visco_wat

class WatRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, wat_sat):
        wat_index = 0
        krw_index = 1
        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            krw = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, krw_index)
        else:
            krw = Table.LinearInterP(self.SWOF, wat_sat, wat_index, krw_index)

        return krw


class OilRelPerm:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, wat_sat):
        wat_index = 0
        kro_index = 2
        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            kro = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, kro_index)
        else:
            kro = Table.LinearInterP(self.SWOF, wat_sat, wat_index, kro_index)

        return kro

class CapillarypressurePcow:
    def __init__(self, pvt):
        super().__init__()
        self.pvt = pvt
        self.SWOF = get_table_keyword(self.pvt, 'SWOF')

    def evaluate(self, wat_sat):
        wat_index = 0
        pc_index = 3
        Table = TableInterpolation()
        if (wat_sat < self.SWOF[0][0] or wat_sat > self.SWOF[len(self.SWOF) - 1][0]):
            pc = Table.SCALExtraP(self.SWOF, wat_sat, wat_index, pc_index)
        else:
            pc = Table.LinearInterP(self.SWOF, wat_sat, wat_index, pc_index)

        return pc

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

