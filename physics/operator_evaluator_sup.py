import numpy as np
from darts.engines import *
from darts.physics import *

import os.path as osp

physics_name = osp.splitext(osp.basename(__file__))[0]

# Define our own operator evaluator class
class ReservoirOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        #interpr.comp_out_of_bounds(zc)

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term: """
        for i in range(nc):
            values[i] = self.compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in self.ph:
            values[shift + j] = 0

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in self.ph:
                values[shift + i * nph + j] = 0

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in self.ph:
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        for i in self.ph:
            values[shift + 3 + nph + i] = self.pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = 0

        # print(state, values)

        return 0

class WellOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

        compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        #interpr.comp_out_of_bounds(zc)

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term: """
        for i in range(nc):
            values[i] = compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in ph:
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = x[j][i] * rho_m[j] * kr[j] / mu[j]

        shift += nph + nph * ne + ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = rho[i]

        return 0

class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.flux = np.zeros(self.nc)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.nph):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)


        self.flux[:] = 0
        # step-1
        for j in ph:
            for i in range(self.nc):
                self.flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(self.flux)

        (sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = sat_sc[j] * flux_sum / total_density

        return 0


# Define our own operator evaluator class
class ReservoirThermalOperators(ReservoirOperators):
    def __init__(self, property_container, thermal=1):
        super().__init__(property_container, thermal=thermal)  # Initialize base-class

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        super().evaluate(state, values)

        vec_state_as_np = np.asarray(state)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        (enthalpy, rock_energy) = self.property.evaluate_thermal(state)

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        i = nc  # use this numeration for energy operators
        """ Alpha operator represents accumulation term: """
        for m in self.ph:
            values[i] += self.compr * self.sat[m] * self.rho_m[m] * enthalpy[m]  # fluid enthalpy (kJ/m3)
        values[i] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = ne + ne * j
            values[shift + i] = enthalpy[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Chi operator for temperature in conduction, gamma operators are skipped """
        shift = ne + ne * nph + nph
        for j in range(nph):
            values[shift + nc * nph + j] = temperature

        """ Delta operator for reaction """
        shift += nph * ne
        values[shift + i] = 0

        """ Additional energy operators """
        shift += ne
        # E1-> rock internal energy
        values[shift] = rock_energy / self.compr  # kJ/m3
        # E2-> rock temperature
        values[shift + 1] = temperature
        # E3-> rock conduction
        values[shift + 2] = 1 / self.compr  # kJ/m3

        print(state, values)

        return 0


class PropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        values[0] = self.sat[0]

        return 0
