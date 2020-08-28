import numpy as np
from darts.engines import *
from darts.physics import *
import os.path as osp
from scipy.interpolate import interp1d

physics_name = osp.splitext(osp.basename(__file__))[0]


# Define our own operator evaluator class
class component_acc_flux_etor(operator_set_evaluator_iface):
    def __init__(self, component_data, bool_trans_upd, physics_type, log_based,
                 bool_debug_mode_on=False, output_operators=False):
        super().__init__()  # Initialize base-class

        # Obtain properties from user input during initialization:
        self.vec_pressure_range_k_values = component_data.vec_pressure_range_k_values
        self.vec_thermo_equi_const_k_water = component_data.vec_thermo_equi_const_k_water
        self.vec_thermo_equi_const_k_co2 = component_data.vec_thermo_equi_const_k_co2
        self.sca_k_caco3 = component_data.sca_k_caco3
        self.sca_tolerance = component_data.sca_tolerance
        self.sca_ref_pres = component_data.sca_ref_pres
        self.sca_density_water_stc = component_data.sca_density_water_stc
        self.sca_compressibility_water = component_data.sca_compressibility_water
        self.sca_density_gas_stc = component_data.sca_density_gas_stc
        self.sca_compressibility_gas = component_data.sca_compressibility_gas
        self.sca_density_solid_stc = component_data.sca_density_solid_stc
        self.sca_compressibility_solid = component_data.sca_compressibility_solid
        self.vec_res_sat_mobile_phases = component_data.vec_res_sat_mobile_phases
        self.vec_brooks_corey_exponents = component_data.vec_brooks_corey_exp
        self.vec_end_point_rel_perm = component_data.vec_end_point_rel_perm
        self.vec_viscosity_mobile_phases = component_data.vec_viscosity_mobile_phases
        self.sca_transmissibility_exp = component_data.sca_transmissibility_exp
        self.bool_debug_mode_on = bool_debug_mode_on
        self.bool_trans_upd = bool_trans_upd
        self.physics_type = physics_type
        self.log_based = log_based
        self.num_comp = component_data.num_comp

        # Extra properties for kinetics:
        self.kin_rate = component_data.kin_rate
        self.min_surf_area = component_data.min_surf_area
        self.order_react = component_data.order_react
        self.wat_molal = component_data.wat_molal
        self.equi_prod = component_data.equi_prod
        self.stoich_matrix = component_data.stoich_matrix
        self.min_comp = component_data.min_comp

        # Diffusion related properties:
        self.diff_coef = component_data.diff_coef

        # Additional properties to be added to self:
        self.vec_liquid_molefrac = np.zeros((self.num_comp,))
        self.vec_vapor_molefrac = np.zeros((self.num_comp,))
        self.vec_solid_molefrac = np.array([1, 0, 0, 0, 0])
        self.vec_phase_molefrac = np.zeros((3,))
        self.vec_composition = np.zeros((self.num_comp,))
        self.vec_fractional_flow_components = np.zeros((self.num_comp,))
        self.vec_saturation_all_phases = np.zeros((3,))
        self.sca_phase_frac_weighted_by_density = 1
        self.sca_total_mobility = 1
        self.vec_nonlin_unknowns = np.zeros((9,))
        self.vec_actual_density = np.zeros((3,))
        self.vec_k_values = np.zeros((2,))
        self.str_state_denoter = '000'
        self.sca_trans_multiplier = 1
        self.output_operators = output_operators

    def compute_residual_lig_vap_equi(self, re_norm_comp, non_lin_unkwns):
        # Variables: X = [x_co2, x_ca, x_co3, x_h2o, y_co2, y_h2o, V*]
        #                   0      1     2      3      4      5    6
        # Allocate memory:
        residual = np.zeros((7,))

        # Equation 0:3 are for scaled composition: z_c* = x_i*(1-V*) + y_i*V*
        residual[:4] = (1 - non_lin_unkwns[6])*non_lin_unkwns[:4] + \
                       non_lin_unkwns[6]*np.concatenate(([non_lin_unkwns[4]], [0, 0], [non_lin_unkwns[5]])) - \
                       re_norm_comp

        # Next two equations are phase equilibrium constraints:
        residual[4:6] = self.vec_k_values*np.concatenate(([non_lin_unkwns[0]], [non_lin_unkwns[3]])) - non_lin_unkwns[4:6]

        # Finally component phase molefrac constraint:
        residual[6] = np.sum(non_lin_unkwns[0:4]) - np.sum(non_lin_unkwns[4:6])
        return residual

    def compute_jacobian_liq_vap_equi(self, re_norm_comp, non_lin_unkwns):
        # Variables: X = [x_co2, x_ca, x_co3, x_h2o, y_co2, y_h2o, V*]
        #                   0     1      2      3      4      5    6
        # Allocate memory:
        jacobian = np.zeros((7, 7))

        # Equation 0:3 are for scaled composition: z_c* = x_i*(1-V*) + y_i*V*
        for ii in range(0, 4):
            # Derivative when component exists in liquid phase:
            jacobian[ii, ii] = 1 - non_lin_unkwns[6]  # w.r.t. x_i
            if ii < 2:
                # Derivative when component exists in vapor phase:
                jacobian[ii, ii+4] = non_lin_unkwns[6]  # w.r.t. y_i
                jacobian[ii, 6] = -non_lin_unkwns[ii] + non_lin_unkwns[ii+4]  # w.r.t. V*
            else:
                # Again, when for liquid phase only components:
                jacobian[ii, 6] = -non_lin_unkwns[ii] + 0  # w.r.t. V*

        # Next two equations are phase equilibrium constraints:
        jacobian[4, 0] = self.vec_k_values[0]  # w.r.t. x_1
        jacobian[5, 3] = self.vec_k_values[1]  # w.r.t. x_2
        jacobian[4, 4] = -1  # w.r.t. y_1
        jacobian[5, 5] = -1  # w.r.t. y_2

        # Finally component phase molefrac constraint:
        jacobian[6, :4] = 1  # w.r.t. x_i
        jacobian[6, 4:6] = -1  # w.r.t. y_i
        return jacobian

    def fun_rachford_rice(self, vapor_molefrac, sca_dum_comp):
        return np.sum(sca_dum_comp * (1 - self.vec_k_values) / (1 + (self.vec_k_values - 1) * vapor_molefrac))

    def two_phase_flash_full_sys(self, sca_dum_comp):
        # sca_dum_comp = np.array([0.4, 0.4, 0.1, 0.1])
        sca_vapor_molefrac_min = 1 / (1 - np.max(self.vec_k_values)) + self.sca_tolerance
        sca_vapor_molefrac_max = 1 / (1 - np.min(self.vec_k_values)) - self.sca_tolerance
        sca_new_vapor_molefrac = (sca_vapor_molefrac_min + sca_vapor_molefrac_max) / 2
        sca_iter_counter = 0
        bisec_conv = 10 ** (-13)
        max_bisec_iter = 50

        while (np.abs(self.fun_rachford_rice(sca_new_vapor_molefrac, sca_dum_comp)) > bisec_conv) and \
                (sca_iter_counter < max_bisec_iter) and \
                (np.abs(sca_new_vapor_molefrac - sca_vapor_molefrac_min) > bisec_conv) and \
                (np.abs(sca_new_vapor_molefrac - sca_vapor_molefrac_max) > bisec_conv):
            # Perform bisection iteration:
            # Check if function is monotonically increasing or decreasing in order to correctly set new interval:
            if (self.fun_rachford_rice(sca_vapor_molefrac_min, sca_dum_comp) *
                    self.fun_rachford_rice(sca_new_vapor_molefrac, sca_dum_comp) < 0):
                sca_vapor_molefrac_max = sca_new_vapor_molefrac

            else:
                sca_vapor_molefrac_min = sca_new_vapor_molefrac

            # Update new interval center value:
            sca_new_vapor_molefrac = (sca_vapor_molefrac_min + sca_vapor_molefrac_max) / 2

            # Increment iteration:
            sca_iter_counter += 1

        # Set vapor and liquid component mole fractions based on physical region:
        if np.abs(self.fun_rachford_rice(sca_new_vapor_molefrac, sca_dum_comp)) < 10 ** (-13):
            two_phase_liq_molefrac = sca_dum_comp / (sca_new_vapor_molefrac * (self.vec_k_values - 1) + 1)
            two_phase_vap_molefrac = self.vec_k_values * two_phase_liq_molefrac
        else:
            print('No converged initial guess found!!!\n')
            two_phase_liq_molefrac = np.zeros((4,))
            two_phase_vap_molefrac = np.zeros((4,))

        # Write output variables similar to other methods:
        # Variables: X = [x_h2o, x_co2, x_ca, x_co3, y_h2o, y_co2, V*]
        #                   0     1      2      3      4      5    6
        sca_new_vapor_molefrac = np.array(sca_new_vapor_molefrac).reshape((1,))
        non_lin_unkwns = np.concatenate((two_phase_liq_molefrac, two_phase_vap_molefrac[:2], sca_new_vapor_molefrac), axis=0)
        return non_lin_unkwns, sca_iter_counter

    def two_phase_flash_newton(self, re_norm_comp):
        # Perform two-phase flash:
        # Initialize nonlinear unknowns:
        non_lin_unkwns = np.array([0.2, 0.05, 0.05, 0.7, 0.75, 0.25, 0.3])
        sca_iter_counter = 0
        newt_conv = 10 ** (-13)
        max_newt_iter = 50

        # Compute residual:
        residual = self.compute_residual_lig_vap_equi(re_norm_comp, non_lin_unkwns)

        while (np.linalg.norm(residual) > newt_conv) and (sca_iter_counter < max_newt_iter):
            # Perform Newton loop:
            # Compute Jacobian:
            jacobian = self.compute_jacobian_liq_vap_equi(re_norm_comp, non_lin_unkwns)

            # Get update:
            non_lin_upd = -np.linalg.solve(jacobian, residual)

            # Update unknowns:
            non_lin_unkwns += non_lin_upd

            # Update residual:
            residual = self.compute_residual_lig_vap_equi(re_norm_comp, non_lin_unkwns)

            # Update counter:
            sca_iter_counter += 1

        if (np.linalg.norm(residual) > newt_conv):
            print('\n------------------------ERROR------------------------')
            print('\t\t\tNO CONVERGENCE FOR FLASH WITH NEWTON REACHED!!!')
            print('------------------------ERROR------------------------\n')
            exit()

        return non_lin_unkwns, sca_iter_counter

    def out_bound_composition_new(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_comp:
                vec_composition[ith_comp] = self.min_comp
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_comp:
                vec_composition[ith_comp] = 1 - self.min_comp
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_comp);
        return vec_composition

    def eval_bounds_nonlin_unkwns(self, vec_nonlin_unknowns):
        """
        Class method which evaluate if the nonlinear uknowns are out of physical bounds
        :param vec_nonlin_unknowns: vector with nonlinear unknowns for Newton-Loop
        :return vec_nonlin_unknowns: "                                          "
        :return temp_index: boolean vector containing true for each phase not present
        """
        # NOTE: nonlin_unknws_full = X = [x_co2, x_co3, x_ca, x_h2o, y_co2, y_h2o, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Check for negative values in the liquid and vapor component fractions as well as phase fractions:
        temp_index = vec_nonlin_unknowns <= self.min_comp
        if temp_index.any():
            # At least one truth value in the boolean array:
            vec_nonlin_unknowns[temp_index] = 0

            # Rescale all variables so that they sum to one:
            vec_nonlin_unknowns[0:4] = vec_nonlin_unknowns[0:4] / np.sum(
                vec_nonlin_unknowns[0:4])
            vec_nonlin_unknowns[4:6] = vec_nonlin_unknowns[4:6] / np.sum(
                vec_nonlin_unknowns[4:6])
            vec_nonlin_unknowns[6:] = vec_nonlin_unknowns[6:] / np.sum(vec_nonlin_unknowns[6:])
        return vec_nonlin_unknowns, temp_index

    def eval_comp(self, vec_nonlin_unknowns):
        """
        Class method which evaluates component total composition
        :return vec_component_comp: vector with component composition
        """
        # NOTE: nonlin_unknws_full = X = [x_co2, x_co3, x_ca, x_h2o, y_co2, y_h2o, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        vec_component_comp = np.concatenate(([0], vec_nonlin_unknowns[0:4])) * vec_nonlin_unknowns[6] + \
                             np.concatenate(([0], [vec_nonlin_unknowns[4]], [0, 0], [vec_nonlin_unknowns[5]])) * vec_nonlin_unknowns[7] + \
                             self.vec_solid_molefrac * vec_nonlin_unknowns[8]
        return vec_component_comp


    def state_denoter(self, temp_index):
        """
        Class method which using information from state to update state denoter
        :param temp_index:
        :return str_state_denoter: updated state denoter on current state of system
        """
        if temp_index[6] and temp_index[7]:
            # No liquid and vapor phase present, therefore in state 100:
            str_state_denoter = '100'
        elif temp_index[6] and temp_index[8]:
            # No liquid and solid phase present, therefore in state: 010:
            str_state_denoter = '010'
        elif temp_index[7] and temp_index[8]:
            # No vapor and solid phase present, therefore in state: 001:
            str_state_denoter = '001'
        elif temp_index[6]:
            # No liquid phase present, therefore at least in state 110:
            str_state_denoter = '110'
        elif temp_index[7]:
            # No vapor phase present, therefore at least in state 101:
            str_state_denoter = '101'
        elif temp_index[8]:
            # No solid phase present, therefore at least in state 011:
            str_state_denoter = '011'
        else:
            # Pure three phase system:
            str_state_denoter = '111'
        return str_state_denoter

    def store_final_solution(self, vec_nonlin_unknowns):
        # Store solution in self for access outside of class
        self.vec_liquid_molefrac = np.concatenate(([0], vec_nonlin_unknowns[:4]))
        self.vec_vapor_molefrac = np.concatenate(([0], [vec_nonlin_unknowns[4]], [0, 0], [vec_nonlin_unknowns[5]]))
        self.vec_phase_molefrac = vec_nonlin_unknowns[6:]
        self.vec_nonlin_unknowns = vec_nonlin_unknowns
        self.vec_composition = self.eval_comp(vec_nonlin_unknowns)

    def eval_wat_density(self, sca_pres):
        return self.sca_density_water_stc*(1 + self.sca_compressibility_water*(sca_pres - self.sca_ref_pres))

    def eval_gas_density(self, sca_pres):
        return self.sca_density_gas_stc * np.exp(self.sca_compressibility_gas * (sca_pres - self.sca_ref_pres))

    def eval_sol_density(self, sca_pres):
        return self.sca_density_solid_stc * (1 + self.sca_compressibility_solid * (sca_pres - self.sca_ref_pres))

    @staticmethod
    def eval_wat_density_with_co2(co2_molefrac, wat_density, gas_density):
        return 1 / (co2_molefrac / gas_density + (1 - co2_molefrac) / wat_density)

    def eval_sat_dens(self, sca_pressure):
        """
        Class method which evaluates the saturation and density of current system, based on state[0]==pressure
        :param sca_pressure: state dependent variables, pressure (state[0])
        :param vec_nonlin_unknowns: full vector of converged nonlinear unknowns
        :return: updated vectors for saturation, density, and weighted saturation by density
        """
        # Evaluate density:
        sca_actual_density_water = self.eval_wat_density(sca_pressure)
        sca_actual_density_gas = self.eval_gas_density(sca_pressure)
        sca_actual_density_solid = self.eval_sol_density(sca_pressure)

        # Calculate water density with co2 dissolved:
        # sca_actual_density_water = element_acc_flux_etor.eval_wat_density_with_co2(self.vec_nonlin_unknowns[1],
        #                                                                            sca_actual_density_water,
        #                                                                            sca_actual_density_gas)

        # Store density of phases in vector:
        self.vec_actual_density = np.array([sca_actual_density_water,
                                            sca_actual_density_gas,
                                            sca_actual_density_solid]).flatten()

        # Determine saturation:
        self.sca_phase_frac_weighted_by_density = np.sum(self.vec_nonlin_unknowns[6:] / self.vec_actual_density)
        self.vec_saturation_all_phases = (self.vec_nonlin_unknowns[6:] / self.vec_actual_density) / \
                                          self.sca_phase_frac_weighted_by_density
        return 0

    def eval_fracflow_comp(self):
        """
        Class method which evaluates the fractional flow for each component, based on full state of system
        :return: updated liquid and vapor molefrac, total mobility coefficient, total composition, and fracflow comp
        """
        # Compute effective saturations for relperm related calculations:
        # Check if mobile phases present:
        if np.sum(self.vec_saturation_all_phases[:-1]) < 10**(-5):
            # If no immobile phase present, set mobile_phase saturation to 0:
            vec_saturation_mobile_phases = np.zeros((2,))
            print('WARNING: NO MOBILE PHASE PRESENT!!!')
        else:
            # First rescale the saturation to only mobile phases:
            vec_saturation_mobile_phases = self.vec_saturation_all_phases[:-1] / \
                                           np.sum(self.vec_saturation_all_phases[:-1])


        # Calculate effective saturations (of mobile phases):
        sca_normalizer_eff_sat = 1 - np.sum(self.vec_res_sat_mobile_phases)
        vec_eff_saturation = (vec_saturation_mobile_phases - self.vec_res_sat_mobile_phases) / sca_normalizer_eff_sat

        # Compute relative permeability:
        vec_relative_permeability = self.vec_end_point_rel_perm * \
                                    (vec_eff_saturation ** self.vec_brooks_corey_exponents)
        vec_mobility = vec_relative_permeability / self.vec_viscosity_mobile_phases
        self.sca_total_mobility = np.sum(vec_mobility)

        # Compute fractional flow for each phase:
        vec_fractional_flow_phases = vec_mobility / self.sca_total_mobility

        # Check for NAN:
        if any(np.isnan(vec_fractional_flow_phases)):
            vec_fractional_flow_phases = np.zeros((2,))

        self.vec_fractional_flow_components = self.vec_liquid_molefrac * vec_fractional_flow_phases[0] * \
                                              self.vec_actual_density[0] + \
                                              self.vec_vapor_molefrac * vec_fractional_flow_phases[1] * \
                                              self.vec_actual_density[1]
        return 0

    def calc_thermodynamic_state(self, sca_pressure, vec_composition):
        """
        Class method which computes the thermodynamic state of the system based on the current state
        :param sca_pressure: scalar value of cell pressure [bar]
        :param vec_composition: vector of cell compositions [comp_0, ..., comp_N-1]
        :return: save converged vector of non-linear unknowns to self
        """
        # Find new K_water and K_co2, the thermodynamic constants:
        # sca_k_water = interp1d(self.vec_pressure_range_k_values, self.vec_thermo_equi_const_k_water)(sca_pressure)
        # sca_k_co2 = interp1d(self.vec_pressure_range_k_values, self.vec_thermo_equi_const_k_co2)(sca_pressure)
        sca_k_water = 0.1080
        sca_k_co2 = 1149
        self.vec_k_values = np.array([sca_k_co2, sca_k_water]).flatten()

        # Get more educated initial guess for three-phase flash:
        solid_composition = vec_composition[0]
        normalized_composition = vec_composition[1:] / (1 - solid_composition)

        # Calculat phase-split with Newton:
        non_lin_unkwns, sca_iter_counter = self.two_phase_flash_newton(normalized_composition)

        # Get re-normalized phase-fractions:
        phase_fractions = np.array([(1 - non_lin_unkwns[6]) * (1 - solid_composition),
                                    non_lin_unkwns[6] * (1 - solid_composition),
                                    solid_composition])

        # Store in correct solution format:
        vec_nonlin_unknowns = np.concatenate((non_lin_unkwns[:6], phase_fractions), axis=0)

        # Check for negative values in the liquid and vapor component fractions as well as phase fractions:
        vec_nonlin_unknowns, temp_index = self.eval_bounds_nonlin_unkwns(vec_nonlin_unknowns)

        # Determine if system is in three-phase or less:
        # Phase denotation system according to AD-GPRS logic for three-phase system:
        # Water = 0, Oil = 1, Gas = 2 || Liquid = 0, Vapor = 1, Solid = 2
        # system_state 000 001 010 100 011 101 110 111
        # phase_0       -   x   -   -   x   x   -   x
        # phase_1       -   -   x   -   x   -   x   x
        # phase_2       -   -   -   x   -   x   x   x
        # Since zero phases present is not possible in our system, there are 2^{n}-1 states possible!
        str_state_denoter = self.state_denoter(temp_index)

        if str_state_denoter == '100':
            print('NO MOBILE PHASE PRESENT!!!!')

        # Store final converged vector of non-linear unknowns, which represent component molefractions in
        # the liquid, vapor, and solid phase, as well as phase mole fractions:
        self.store_final_solution(vec_nonlin_unknowns)
        return 0

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, elem_comp_0, ..., elem_comp_N-1]
        :param values: values of the operators
        :return: updated value for operators, stored in values
        """
        # Composition vector:
        vec_state_as_np = np.asarray(state)

        if self.log_based == 1:
            vec_composition = np.append(np.exp(vec_state_as_np[1:]), 1 - np.sum(np.exp(vec_state_as_np[1:])))
        else:
            vec_composition = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        vec_comp_copy = np.array(vec_composition, copy=True)
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 - self.min_comp or element comp below self.min_comp, i.e. if point is unphysical or
        # out of OBL domain:
        vec_composition = self.out_bound_composition_new(vec_composition)

        # -----------
        # STEP 1, 2, 3:
        # -----------
        self.calc_thermodynamic_state(sca_pressure, vec_composition)

        # --------------------------------------------------------
        # STEP 4: Compute all state related variables:
        # --------------------------------------------------------
        # Now phase state fully defined, general way of calculating density, saturation, relative permeability, etc.:
        self.eval_sat_dens(sca_pressure)

        # Compute fractional flow of each component, the total mobility scalar, and the total composition:
        self.eval_fracflow_comp()

        # Compute transmissibility multiplier:
        if self.bool_trans_upd:
            self.sca_trans_multiplier = (1 - self.vec_saturation_all_phases[2]) ** self.sca_transmissibility_exp

        # --------------------------------------------------------
        # STEP 5: Compute all state dependent operators:
        # --------------------------------------------------------
        sca_total_comp_density = 1 / self.sca_phase_frac_weighted_by_density
        sca_rock_compres_factor = 1 + self.sca_compressibility_solid * (sca_pressure - self.sca_ref_pres)

        # Alpha operator:
        num_alpha_op = self.num_comp
        for ith_alpha in range(num_alpha_op):
            values[ith_alpha] = sca_rock_compres_factor * sca_total_comp_density * vec_comp_copy[ith_alpha]

        # Beta operator:
        num_beta_op = self.num_comp
        temp_beta_operator = self.sca_trans_multiplier * self.sca_total_mobility * self.vec_fractional_flow_components

        for ith_beta in range(num_beta_op):
            values[ith_beta + num_alpha_op] = temp_beta_operator[ith_beta]

        # Allocate number of kinetic and diffusion operators:
        num_kin_op = 0
        num_diff_op = 0

        if self.physics_type == 'kinetics' or self.physics_type == 'kin_diff':
            # Store each kinetic operator:
            sol_prod = self.wat_molal ** 2 * self.vec_liquid_molefrac[2] * self.vec_liquid_molefrac[3] / (
                        self.vec_liquid_molefrac[-1] ** 2)

            # NEW WAY:
            num_kin_op = self.num_comp
            for ith_gamma in range(num_kin_op):
                values[ith_gamma + num_alpha_op + num_beta_op] = \
                    -self.min_surf_area * self.kin_rate * (1 - (sol_prod/self.equi_prod)**self.order_react) * \
                    self.stoich_matrix[ith_gamma] * (vec_comp_copy[0] - self.min_comp) * sca_total_comp_density

        if self.physics_type == 'kin_diff' or self.physics_type == 'diffusion':
            # Store each diffusion operator:
            num_diff_op = self.num_comp
            for ith_delta in range(num_diff_op):
                # values[ith_delta + num_alpha_op + num_beta_op + num_kin_op] = self.diff_coef[ith_delta] * vec_composition[ith_delta]
                values[ith_delta + num_alpha_op + num_beta_op + num_kin_op] = self.diff_coef[ith_delta] * values[ith_delta]

        if self.physics_type == 'kinetics' or self.physics_type == 'kin_diff':
            # Store porosity operator:
            values[num_alpha_op + num_beta_op + num_kin_op + num_diff_op] = 1 - vec_comp_copy[0]

        # if np.sum(np.exp(state[1])) > 0.99:
        #     print('\n------------------------ERROR------------------------')
        #     print('\t\t\tNO FLUID PHASE FOUND!!!')
        #     print('------------------------ERROR------------------------\n')
        #     # exit()
        # elif any(np.isnan(values)):
        #     print('\n------------------------ERROR------------------------')
        #     print('\t\t\tNAN Returned for values!!!')
        #     print('------------------------ERROR------------------------\n')
        #     # exit()
        return 0


class component_acc_flux_data():
    """
    This class holds all the necessary (user)input data for evaluating the accumulation, flux, and other properties
    for the elements based physics
    """
    def __init__(self, vec_pressure_range_k_values, vec_thermo_equi_const_k_water,
                       vec_thermo_equi_const_k_co2, sca_k_caco3, sca_tolerance,
                       sca_ref_pres, sca_density_water_stc, sca_compressibility_water, sca_density_gas_stc,
                       sca_compressibility_gas, sca_density_solid_stc, sca_compressibility_solid,
                       vec_res_sat_mobile_phases, vec_brooks_corey_exp, vec_end_point_rel_perm,
                       vec_viscosity_mobile_phases, sca_transmissibility_exp, num_comp, min_comp, kin_data):
        # Assign data to data structure:
        self.vec_pressure_range_k_values = vec_pressure_range_k_values
        self.vec_thermo_equi_const_k_water = vec_thermo_equi_const_k_water
        self.vec_thermo_equi_const_k_co2 = vec_thermo_equi_const_k_co2
        self.sca_k_caco3 = sca_k_caco3
        self.sca_tolerance = sca_tolerance
        self.sca_ref_pres = sca_ref_pres
        self.sca_density_water_stc = sca_density_water_stc
        self.sca_compressibility_water = sca_compressibility_water
        self.sca_density_gas_stc = sca_density_gas_stc
        self.sca_compressibility_gas = sca_compressibility_gas
        self.sca_density_solid_stc = sca_density_solid_stc
        self.sca_compressibility_solid = sca_compressibility_solid
        self.vec_res_sat_mobile_phases = vec_res_sat_mobile_phases
        self.vec_brooks_corey_exp = vec_brooks_corey_exp
        self.vec_end_point_rel_perm = vec_end_point_rel_perm
        self.vec_viscosity_mobile_phases = vec_viscosity_mobile_phases
        self.sca_transmissibility_exp = sca_transmissibility_exp
        self.kin_rate = kin_data.kin_rate
        self.min_surf_area = kin_data.min_surf_area
        self.order_react = kin_data.order_react
        self.wat_molal = kin_data.wat_molal
        self.equi_prod = kin_data.equi_prod
        self.num_comp = num_comp
        self.stoich_matrix = kin_data.stoich_matrix
        self.diff_coef = kin_data.diff_coef
        self.min_comp = min_comp


class chemical_rate_evaluator(component_acc_flux_etor):
    # Simplest class existing to mankind:
    def __init__(self, elements_data, bool_trans_upd, physics_type, log_based,
                 bool_debug_mode_on=False, output_operators=False):
        # Initialize base-class
        super().__init__(elements_data, bool_trans_upd, physics_type, log_based,
                         bool_debug_mode_on, output_operators)

        # Add mobility vector to class:
        self.vec_mobility = np.zeros((2,))
        self.vec_saturation_mobile_phases = np.zeros((2,))

    def calc_mobility(self):
        """
        Class method which evaluates the fractional flow for each component, based on full state of system
        :return: updated liquid and vapor molefrac, total mobility coefficient, total composition, and fracflow comp
        """
        # Compute effective saturations for relperm related calculations:
        # Check if mobile phases present:
        if np.sum(self.vec_saturation_all_phases[:-1]) < 10 ** (-5):
            # If any immobile phase present, set mobile_phase saturation to 0:
            self.vec_saturation_mobile_phases = np.zeros((2,))
            print('WARNING: NO MOBILE PHASE PRESENT!!!')
        else:
            # First rescale the saturation to only mobile phases:
            self.vec_saturation_mobile_phases = self.vec_saturation_all_phases[:-1] / \
                                           np.sum(self.vec_saturation_all_phases[:-1])

        # Calculate effective saturations (of mobile phases):
        sca_normalizer_eff_sat = 1 - np.sum(self.vec_res_sat_mobile_phases)
        vec_eff_saturation = (self.vec_saturation_mobile_phases - self.vec_res_sat_mobile_phases) / sca_normalizer_eff_sat

        # Compute relative permeability:
        vec_relative_permeability = self.vec_end_point_rel_perm * \
                                    (vec_eff_saturation ** self.vec_brooks_corey_exponents)
        self.vec_mobility = vec_relative_permeability / self.vec_viscosity_mobile_phases

    def evaluate(self, state, values):
        """
        Class methods which computes the rate evaluator for a particular state
        :param state: current thermodynamical state of the system (vector with pressure and element compositions)
        :param values: vector with the operator values (each value is the volumetric flow rate of a phase)
        :return: update values of the rate operator based on the current state
        """
        # For state at current iteration, the piece of code will do the following steps:
        #   1) Initialize unknowns (to be solving in physical region)
        #   2) Compute three-phase flash
        #   3) Determine if in 3-phase or not
        #       3.1) If 3-phase proceeed to 4
        #       3.2) If not 3-phase, compute 2-phase flash
        #   4) Based on particular phase-state (7 total), compute all state related variables
        #       such as saturation, relative permeability, etc.
        #   5) Compute operator values
        # Composition vector:
        vec_state_as_np = np.asarray(state)

        if self.log_based == 1:
            vec_composition = np.append(np.exp(vec_state_as_np[1:]), 1 - np.sum(np.exp(vec_state_as_np[1:])))
        else:
            vec_composition = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        vec_comp_copy = np.array(vec_composition, copy=True)
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 - min_comp or element comp below min_comp, i.e. if point is unphysical or
        # out of OBL domain:
        vec_composition = self.out_bound_composition_new(vec_composition)

        # -----------
        # STEP 1, 2, 3:
        # -----------
        self.calc_thermodynamic_state(sca_pressure, vec_composition)

        # --------------------------------------------------------
        # STEP 4: Compute all state related variables:
        # --------------------------------------------------------
        # Now phase state fully defined, general way of calculating density, saturation, relative permeability, etc.:
        self.eval_sat_dens(sca_pressure)

        # Compute fractional flow of each component, the total mobility scalar, and the total composition:
        self.calc_mobility()

        # Calculate molar flux for each component:
        vec_molar_flux_water = self.vec_nonlin_unknowns[0:4] * self.vec_actual_density[0] * self.vec_mobility[0]
        vec_molar_flux_vapor = self.vec_nonlin_unknowns[4:6] * self.vec_actual_density[1] * self.vec_mobility[1]
        vec_total_molar_flux = np.sum(vec_molar_flux_water) + np.sum(vec_molar_flux_vapor)

        # Get total density:
        sca_total_density = np.sum(self.vec_saturation_mobile_phases * self.vec_actual_density[:-1])

        # Easiest example, constant volumetric phase rate:
        values[0] = self.vec_saturation_mobile_phases[1] * vec_total_molar_flux / sca_total_density   # vapor phase
        values[1] = self.vec_saturation_mobile_phases[0] * vec_total_molar_flux / sca_total_density   # liquid phase

        # Usually some steps will be executed to estimate unit volumetric flow rate based on current state (when
        # multiplied with a pressure difference one obtains actual volumetric flow rate)
        return 0
