import numpy as np
from darts.physics import *
from darts.tools.interpolation import TableInterpolation

class dead_oil_table_density_evaluator(property_evaluator_iface):
     def __init__(self, pvdo, surface_oil_dens):
        super().__init__()
        self.oil_dens = surface_oil_dens
        self.table = pvdo

     def evaluate(self, state):
        pres = state[0]
        pres_index = 0             # first column in the table - pressure
        Bo_index   = 1             # second column in the table - volume factor
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            Bo = Table.LinearExtraP(self.table, pres, pres_index, Bo_index)
        else:
            Bo = Table.LinearInterP(self.table, pres, pres_index, Bo_index)

        return self.oil_dens / Bo


class dead_oil_string_density_evaluator(property_evaluator_iface):
    def __init__(self, pvtw, surface_wat_dens):
        super().__init__()
        self.wat_dens = surface_wat_dens
        self.table = pvtw

    def evaluate(self, state):
        pres = state[0]
        X = self.table[2] * (pres - self.table[0])
        Bw = self.table[1] / (1 + X + X * X / 2)

        return self.wat_dens / Bw


class dead_oil_table_viscosity_evaluator(property_evaluator_iface):
    def __init__(self, pvdo):
        super().__init__()
        self.table = pvdo

    def evaluate(self, state):
        pres = state[0]
        pres_index = 0  # first column in the table - pressure
        muo_index = 2   # third column in the table - viscosity
        Table = TableInterpolation()

        if (pres < self.table[0][0] or pres > self.table[len(self.table) - 1][0]):
            muo = Table.LinearExtraP(self.table, pres, pres_index, muo_index)
        else:
            muo = Table.LinearInterP(self.table, pres, pres_index, muo_index)

        return muo

class dead_oil_string_viscosity_evaluator(property_evaluator_iface):
    def __init__(self, pvtw):
        super().__init__()
        self.table = pvtw

    def evaluate(self, state):
        pres = state[0]
        Y = -self.table[4] * (pres - self.table[0]);
        muw = self.table[3] / (1 + Y + Y * Y / 2);

        return muw

class dead_oil_water_saturation_evaluator(property_evaluator_iface):
    def __init__(self, pvdo, pvtw, surface_oil_dens, surface_water_dens):
        super().__init__()
        self.pvdo = pvdo
        self.pvtw = pvtw
        self.surface_oil_dens = surface_oil_dens
        self.surface_wat_dens = surface_water_dens

    def evaluate(self, state):
        wat_composition = state[1]
        water_density = dead_oil_string_density_evaluator(self.pvtw, self.surface_wat_dens)
        wat_dens = water_density.evaluate(state)
        oil_density = dead_oil_table_density_evaluator(self.pvdo, self.surface_oil_dens)
        oil_dens = oil_density.evaluate(state)
        water_sat = wat_composition * oil_dens / (wat_composition * oil_dens + wat_dens - wat_composition * wat_dens)
        water_sat = np.max([water_sat, 0])
        water_sat = np.min([water_sat, 1.0])

        return water_sat

class table_phase1_relative_permeability_evaluator(property_evaluator_iface):
    def __init__(self, swof, pvdo, pvtw, surface_oil_dens, surface_water_dens):
        super().__init__()
        self.table  = swof       # relperm table
        self.pvdo = pvdo
        self.pvtw = pvtw
        self.surface_oil_dens = surface_oil_dens
        self.surface_wat_dens = surface_water_dens

    def evaluate(self, state):
        water_saturation = dead_oil_water_saturation_evaluator(self.pvdo, self.pvtw, self.surface_oil_dens, self.surface_wat_dens)
        wat_sat = water_saturation.evaluate(state)
        wat_index = 0
        krw_index = 1
        Table = TableInterpolation()
        if (wat_sat < self.table[0][0] or wat_sat > self.table[len(self.table) - 1][0]):
            krw = Table.SCALExtraP(self.table, wat_sat, wat_index, krw_index)
        else:
            krw = Table.LinearInterP(self.table, wat_sat, wat_index, krw_index)

        return krw


class table_phase2_relative_permeability_evaluator(property_evaluator_iface):
    def __init__(self, swof, pvdo, pvtw, surface_oil_dens, surface_water_dens):
        super().__init__()
        self.table = swof  # relperm table
        self.pvdo = pvdo
        self.pvtw = pvtw
        self.surface_oil_dens = surface_oil_dens
        self.surface_wat_dens = surface_water_dens

    def evaluate(self, state):
        water_saturation = dead_oil_water_saturation_evaluator(self.pvdo, self.pvtw, self.surface_oil_dens, self.surface_wat_dens)
        wat_sat = water_saturation.evaluate(state)
        wat_index = 0
        kro_index = 2
        Table = TableInterpolation()
        if (wat_sat < self.table[0][0] or wat_sat > self.table[len(self.table) - 1][0]):
            kro = Table.SCALExtraP(self.table, wat_sat, wat_index, kro_index)
        else:
            kro = Table.LinearInterP(self.table, wat_sat, wat_index, kro_index)

        return kro

class table_phase_capillary_pressure_evaluator(property_evaluator_iface):
    def __init__(self, swof, pvdo, pvtw, surface_oil_dens, surface_water_dens):
        super().__init__()
        self.table = swof   # relperm table
        self.pvdo = pvdo
        self.pvtw = pvtw
        self.surface_oil_dens = surface_oil_dens
        self.surface_wat_dens = surface_water_dens

    def evaluate(self, state):
        water_saturation = dead_oil_water_saturation_evaluator(self.pvdo, self.pvtw, self.surface_oil_dens, self.surface_wat_dens)
        wat_sat = water_saturation.evaluate(state)
        wat_index = 0
        pc_index = 3
        Table = TableInterpolation()
        if (wat_sat < self.table[0][0] or wat_sat > self.table[len(self.table) - 1][0]):
            pc = Table.SCALExtraP(self.table, wat_sat, wat_index, pc_index)
        else:
            pc = Table.LinearInterP(self.table, wat_sat, wat_index, pc_index)

        return pc

class custom_rock_compaction_evaluator(property_evaluator_iface):
    def __init__(self, rock):
        super().__init__()
        self.rock_table = rock

    def evaluate(self, state):
        pressure = state[0]
        pressure_ref = self.rock_table[0][0]
        compressibility = self.rock_table[0][1]
        return (1.0 + compressibility * (pressure - pressure_ref))

