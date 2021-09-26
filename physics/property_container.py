import numpy as np

class property_container:
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.nph = len(phases_name)
        self.nc = len(components_name)
        self.components_name = components_name
        self.phases_name = phases_name
        self.min_z = min_z
        self.Mw = Mw

        self.rock_comp = 4.35e-5
        self.p_ref = 277.0

        # Allocate (empty) evaluators for functions
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.enthalpy_ev = []
        self.rock_energy_ev = []
        self.capillary_pressure_ev = []
        self.flash_ev = 0

        # passing arguments
        self.x = np.zeros((self.nph, self.nc))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.sat = np.zeros(self.nph)
        self.nu = np.zeros(self.nph)
        self.mu = np.zeros(self.nph)
        self.kr = np.zeros(self.nph)
        self.pc = np.zeros(self.nph)
        self.enthalpy = np.zeros(self.nph)

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy]


    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def clean_arrays(self):
        for a in self.phase_props:
            a[:] = 0
        for j in range(self.nph):
            self.x[j][:] = 0

    def compute_saturation(self, ph):
        if len(ph) == 1:
            self.sat[ph[0]] = 1
        elif len(ph) == 2:
            denom = self.dens_m[ph[0]] - self.dens_m[ph[0]] * self.nu[ph[0]] + self.dens_m[ph[1]] * self.nu[ph[0]]
            self.sat[ph[0]] = self.dens_m[ph[1]] * self.nu[ph[0]] / denom
            self.sat[ph[1]] = self.dens_m[ph[0]] * self.nu[ph[1]] / denom
        else:
            denom = self.dens_m[0] * self.dens_m[1] * self.nu[2] + self.dens_m[0] * self.dens_m[2] * self.nu[1]\
                  + self.dens_m[1] * self.dens_m[2] * self.nu[0]
            self.sat[0] = self.dens_m[1] * self.dens_m[2] * self.nu[0] / denom
            self.sat[1] = self.dens_m[0] * self.dens_m[2] * self.nu[1] / denom
            self.sat[2] = self.dens_m[0] * self.dens_m[1] * self.nu[2] / denom

    def run_flash(self, pressure, zc):

        (self.x, self.nu) = self.flash_ev.evaluate(pressure, zc)

        ph = []
        for j in range(self.nph):
            if self.nu[j] > 0:
                ph.append(j)

        return ph



    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last

        ph = self.run_flash(pressure, zc)

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, self.x[j][0])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_thermal(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        temperature = vec_state_as_np[-1]

        for m in range(self.nph):
            self.enthalpy[m] = self.enthalpy_ev[self.phases_name[m]].evaluate(temperature)

        rock_energy = self.rock_energy_ev.evaluate(temperature)

        return self.enthalpy, rock_energy

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        ph = self.run_flash(pressure, zc)

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, self.x[j][0]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m

