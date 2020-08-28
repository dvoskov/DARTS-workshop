import numpy as np
from darts.engines import *
from darts.physics import *
import os.path as osp
from scipy.interpolate import interp1d

physics_name = osp.splitext(osp.basename(__file__))[0]


# Define our own operator evaluator class
class element_acc_flux_etor(operator_set_evaluator_iface):
    def __init__(self, elements_data, bool_trans_upd=True, bool_debug_mode_on=False, output_operators=False):
        super().__init__()  # Initialize base-class

        # Obtain properties from user input during initialization:
        self.mat_rate_annihilation = elements_data.mat_rate_annihilation
        self.vec_pressure_range_k_values = elements_data.vec_pressure_range_k_values
        self.vec_thermo_equi_const_k_water = elements_data.vec_thermo_equi_const_k_water
        self.vec_thermo_equi_const_k_co2 = elements_data.vec_thermo_equi_const_k_co2
        self.sca_k_caco3 = elements_data.sca_k_caco3
        self.sca_tolerance = elements_data.sca_tolerance
        self.sca_ref_pres = elements_data.sca_ref_pres
        self.sca_density_water_stc = elements_data.sca_density_water_stc
        self.sca_compressibility_water = elements_data.sca_compressibility_water
        self.sca_density_gas_stc = elements_data.sca_density_gas_stc
        self.sca_compressibility_gas = elements_data.sca_compressibility_gas
        self.sca_density_solid_stc = elements_data.sca_density_solid_stc
        self.sca_compressibility_solid = elements_data.sca_compressibility_solid
        self.vec_res_sat_mobile_phases = elements_data.vec_res_sat_mobile_phases
        self.vec_brooks_corey_exponents = elements_data.vec_brooks_corey_exp
        self.vec_end_point_rel_perm = elements_data.vec_end_point_rel_perm
        self.vec_viscosity_mobile_phases = elements_data.vec_viscosity_mobile_phases
        self.sca_transmissibility_exp = elements_data.sca_transmissibility_exp
        self.bool_debug_mode_on = bool_debug_mode_on
        self.bool_trans_upd = bool_trans_upd
        self.output_operators = output_operators
        self.min_comp = elements_data.min_comp

        # Additional properties to be added to self (initialization of properties):
        self.vec_liquid_molefrac = np.zeros((5,))
        self.vec_vapor_molefrac = np.zeros((5,))
        self.vec_solid_molefrac = np.array([0, 0, 0, 0, 1])
        self.vec_phase_molefrac = np.zeros((3,))
        self.vec_composition = np.zeros((5,))
        self.vec_fractional_flow_components = np.zeros((5,))
        self.vec_saturation_all_phases = np.zeros((3,))
        self.sca_phase_frac_weighted_by_density = 1
        self.sca_total_mobility = 1
        self.vec_nonlin_unknowns = np.zeros((9,))
        self.vec_actual_density = np.zeros((3,))
        self.vec_k_values = np.zeros((2,))
        self.str_state_denoter = '000'
        self.sca_trans_multiplier = 1

    # Some class methods:
    def compute_residual_full_system(self, vec_element_comp, vec_component_comp, vec_nonlin_unknowns):
        """
        Class method which constucts the Residual equations for the full system, containing equations for:
        - component to element mole conservation
        - phase equilibrium equations (liq-vap-sol)
        - chemical equilibrium equations (dissolution CaCO3)
        :param vec_element_comp: element composition (depends on state)
        :param vec_component_comp: component composition (depends on state)
        :param vec_nonlin_unknowns: vector of nonlinear unknowns
        :return: set of residual equations for full system
        """
        # NOTE: nonlin_unknws_full = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Compute residual vector for full set of equations:
        residual = np.zeros((9,))

        # First three equation are z_e*sum{E*z_c} - E*z_c = 0
        residual[0:3] = vec_element_comp[0:-1] * np.sum(np.dot(self.mat_rate_annihilation, vec_component_comp)) - \
                        np.dot(self.mat_rate_annihilation[:-1, :], vec_component_comp)

        # Fourth (Python==3) Equation for sum of composition is zero (1 - sum{z_c} = 0):
        residual[3] = 1 - np.sum(vec_component_comp)

        # Fifth and Sixth Equation are fugacity constraints for water and co2 fractions (Kx - y = 0):
        residual[4:6] = self.vec_k_values * vec_nonlin_unknowns[0:2] - vec_nonlin_unknowns[4:6]

        # Seventh Equation is chemical equilibrium (K - Q = 0)
        # NOTE: Only one chemical equilibrium reaction, invovling Ca+2 and CO3-2 meaning that if initial and boundary
        # conditions for Ca+2 and CO3-2 are the same, they will remain through the simulation:
        # More general case would be:
        #   residual[6] = (55.508**2) * vec_liquid_molefrac_full[2] * vec_liquid_molefrac_full[3]  -
        #                       sca_k_caco3 * (vec_liquid_molefrac_full[0]**2)
        residual[6] = 55.508 * vec_nonlin_unknowns[2] - np.sqrt(self.sca_k_caco3) * vec_nonlin_unknowns[0]

        # Eighth Equation is the sum of the phase fractions equal to 1 (1 - sum{nu_i} = 0):
        residual[7] = 1 - np.sum(vec_nonlin_unknowns[6:])

        # Ninth Equation is the sum of liquid - vapor component fraction should be zero (sum{x_i - y_i} = 0):
        residual[8] = np.sum(vec_nonlin_unknowns[0:4]) - np.sum(vec_nonlin_unknowns[4:6])
        return residual

    def compute_jacobian_full_system(self, vec_element_comp, vec_nonlin_unknowns):
        """
        Class method which constucts the Jacobian matrix for the full system, containing equations for:
        - component to element mole conservation
        - phase equilibrium equations (liq-vap-sol)
        - chemical equilibrium equations (dissolution CaCO3)
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknowns: vector of nonlinear unknowns
        :return: Jacobian matrix of partial derivatives of residual w.r.t. nonlinear unknowns
        """
        # NOTE: nonlin_unknws_full = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Compute jacobian matrix for full set of equations:
        jacobian = np.zeros((9, 9))

        # Calculate sum of the annihilation matrix's columns:
        sum_colums_anni_mat = np.sum(self.mat_rate_annihilation, axis=0)

        # -----------------------------------------------------------------------------------------
        # Compute first three rows of jacobian matrix (res: z_e*sum{E*z_c} - E*z_c = 0):
        # -----------------------------------------------------------------------------------------
        # First four columns w.r.t. liquid component mole fractions:
        for ithCol in range(0, 4):
            jacobian[0:3, ithCol] = vec_element_comp[0:-1] * sum_colums_anni_mat[ithCol] * vec_nonlin_unknowns[6] - \
                                    self.mat_rate_annihilation[0:3, ithCol] * vec_nonlin_unknowns[6]
        # Following two columns w.r.t. vapor component mole fractions:
        for ithCol in range(4, 6):
            jacobian[0:3, ithCol] = vec_element_comp[0:-1] * sum_colums_anni_mat[ithCol - 4] * \
                                    vec_nonlin_unknowns[7] - \
                                    self.mat_rate_annihilation[0:3, ithCol - 4] * vec_nonlin_unknowns[7]

        # Following three columns w.r.t. phase mole fractions:
        jacobian[0:3, 6] = vec_element_comp[0:-1] * np.dot(sum_colums_anni_mat[0:4], vec_nonlin_unknowns[0:4]) - \
                           np.dot(self.mat_rate_annihilation[0:3, 0:4], vec_nonlin_unknowns[0:4])  # w.r.t. liquid phase frac
        jacobian[0:3, 7] = vec_element_comp[0:-1] * np.dot(sum_colums_anni_mat[0:2], vec_nonlin_unknowns[4:6]) - \
                           np.dot(self.mat_rate_annihilation[0:3, 0:2], vec_nonlin_unknowns[4:6])  # w.r.t vapor phase frac
        jacobian[0:3, 8] = vec_element_comp[0:-1] * sum_colums_anni_mat[4] - self.mat_rate_annihilation[0:3, 4]  # solid frac

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for overall composition constraint (1 - sum{z_c} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[3, 0:4] = -vec_nonlin_unknowns[6]
        jacobian[3, 4:6] = -vec_nonlin_unknowns[7]
        jacobian[3, 6] = -np.sum(vec_nonlin_unknowns[0:4])
        jacobian[3, 7] = -np.sum(vec_nonlin_unknowns[4:6])
        jacobian[3, 8] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for fugacity constraints for water and co2 fractions (Kx - y = 0):
        # -----------------------------------------------------------------------------------------
        # k_h2o*x_h2o - y_h2o = 0:
        jacobian[4, 0] = self.vec_k_values[0]
        jacobian[4, 4] = -1
        # k_co2*x_co2 - y_co2 = 0:
        jacobian[5, 1] = self.vec_k_values[1]
        jacobian[5, 5] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for chemical equilibrium constraint (K - Q = 0):
        # -----------------------------------------------------------------------------------------
        # NOTE: See computation of residual, this is not the general case!!!!
        jacobian[6, 0] = -np.sqrt(self.sca_k_caco3)
        jacobian[6, 2] = 55.508

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for phase fractions equal to 1 (1 - sum{nu_i} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[7, 6:] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for sum of liquid - vapor component fraction should be zero (sum{x_i - y_i} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[8, 0:4] = 1
        jacobian[8, 4:6] = -1
        return jacobian

    # Define here other functions that can/should also be called outside of just evaluate, for instance when want to
    # flash the solution:
    def out_bound_composition(self, vec_element_comp):
        """
        Class method which computes if element total composition is out of physical bounds
        :param vec_element_comp: element composition (depends on state)
        :return: physical elemental composition vector
        """
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_index = vec_element_comp <= self.min_comp
        if temp_index.any():
            # At least one truth value in the boolean array:
            vec_element_comp[temp_index] = self.min_comp

        return (vec_element_comp / np.sum(vec_element_comp))

    def fun_rachford_rice(self, vapor_molefrac, sca_dum_comp):
        return np.sum(sca_dum_comp * (1 - self.vec_k_values) / (1 + (self.vec_k_values - 1) * vapor_molefrac))

    def init_three_phase_flash(self):
        """
        Class method which computes an initial guess for the three phase flash, using a two phase (liq-vap) flash
        :return: an intial guess for liquid and vapor comp molefrac
        """
        # Do two-phase flash to have more stable initial guess for three-phase flash:
        # Perform two phase flash to initialize liquid and vapor component mole fraction in the physical region:
        sca_dum_comp = np.array([0.5, 0.5])
        sca_vapor_molefrac_min = 1 / (1 - np.max(self.vec_k_values)) + self.sca_tolerance
        sca_vapor_molefrac_max = 1 / (1 - np.min(self.vec_k_values)) - self.sca_tolerance
        sca_new_vapor_molefrac = (sca_vapor_molefrac_min + sca_vapor_molefrac_max) / 2
        sca_iter_counter = 0

        while (np.abs(self.fun_rachford_rice(sca_new_vapor_molefrac, sca_dum_comp)) > 10 ** (-13)) and sca_iter_counter < 50 \
                and (np.abs(sca_new_vapor_molefrac - sca_vapor_molefrac_min) > 10 ** (-13)) \
                and (np.abs(sca_new_vapor_molefrac - sca_vapor_molefrac_max) > 10 ** (-13)):
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
            two_phase_liq_molefrac = np.array([0.7, 0.3])
            two_phase_vap_molefrac = np.array([0.3, 0.7])
        return two_phase_liq_molefrac, two_phase_vap_molefrac

    def eval_bounds_nonlin_unkwns(self, vec_nonlin_unknowns):
        """
        Class method which evaluate if the nonlinear uknowns are out of physical bounds
        :param vec_nonlin_unknowns: vector with nonlinear unknowns for Newton-Loop
        :return vec_nonlin_unknowns: "                                          "
        :return temp_index: boolean vector containing true for each phase not present
        """
        # Check for negative values in the liquid and vapor component fractions as well as phase fractions:
        temp_index = vec_nonlin_unknowns <= self.min_comp
        if temp_index.any():
            # At least one truth value in the boolean array:
            vec_nonlin_unknowns[temp_index] = self.min_comp

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
        vec_component_comp = np.append(vec_nonlin_unknowns[0:4], [0]) * vec_nonlin_unknowns[6] + \
                             np.append(vec_nonlin_unknowns[4:6], [0, 0, 0]) * vec_nonlin_unknowns[7] + \
                             self.vec_solid_molefrac * vec_nonlin_unknowns[8]
        return vec_component_comp

    def construct_init_nonlinear_unknowns(self, two_phase_liq_molefrac, two_phase_vap_molefrac):
        """
        Class methods which constructs the initial vector of nonlinear unknowns according:
        # NOTE: vec_nonlin_unknowns = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        Based on initial guess!
        :param two_phase_liq_molefrac: initial guess in physical region for liquid component molefractions
        :param two_phase_vap_molefrac: initial guess in physical region for vapor component molefractions
        :return vec_nonlin_unknowns: vector with nonlinear unknowns used in nonlinear Newton loop
        """
        vec_liquid_molefrac = np.append(two_phase_liq_molefrac, [self.min_comp, self.min_comp, 0])
        vec_vapor_molefrac = np.append(two_phase_vap_molefrac, [0, 0, 0])
        vec_phase_molefrac = np.array([0.5, 0.5, self.min_comp])
        vec_nonlin_unknowns = np.append(np.append(vec_liquid_molefrac[0:-1], vec_vapor_molefrac[0:2]),
                                        vec_phase_molefrac)
        return vec_nonlin_unknowns

    def three_phase_flash(self, vec_element_comp, vec_nonlin_unknowns):
        """
        Class method which computes the three-phase flash equilibrium (liq-vap-sol):
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknowns: vector with nonlinear unknowns used in nonlinear Newton loop
        :return vec_nonlin_unknowns: converged solution to nonlinear problem
        :return index: boolean vector, containing true for each phase not present
        """
        # Compute composition:
        vec_component_comp = self.eval_comp(vec_nonlin_unknowns)

        # Compute residual
        vec_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknowns)

        # Start Newton loop to find root for full system of equations:
        sca_iter_counter = 0
        sca_max_iter = 100
        temp_tolerance = self.sca_tolerance
        while (np.linalg.norm(vec_residual) > temp_tolerance) and (sca_iter_counter <= sca_max_iter):
            # Compute Jacobian, used every Newton iteration to compute solution to nonlinear system:
            mat_jacobian = self.compute_jacobian_full_system(vec_element_comp, vec_nonlin_unknowns)

            # Solve linear system:
            vec_nonlin_update = -np.linalg.solve(mat_jacobian, vec_residual)

            # Update non-linear unknowns:
            vec_nonlin_unknowns = vec_nonlin_unknowns + vec_nonlin_update

            # Recompute composition:
            vec_component_comp = self.eval_comp(vec_nonlin_unknowns)

            # Recompute residual equations (vec_residual --> 0 when converged):
            vec_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknowns)

            # Increment iteration counter:
            sca_iter_counter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (sca_iter_counter == 20) or (sca_iter_counter == 40):
                temp_tolerance = temp_tolerance * 10

        if sca_iter_counter > sca_max_iter:
            # NOT CONVERGED:
            print('------------------------WARNING------------------------')
            print('\t\t\t Three-phase equilibrium did not converge')
            print('------------------------WARNING------------------------')

        # Check for negative values in the liquid and vapor component fractions as well as phase fractions:
        vec_nonlin_unknowns, temp_index = self.eval_bounds_nonlin_unkwns(vec_nonlin_unknowns)
        return vec_nonlin_unknowns, temp_index

    def two_phase_vap_sol(self, vec_element_comp, vec_nonlin_unknws_full, str_state_denoter):
        """
        Class method which solves a two-phase flash for vapor-solid equilibrium
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknws_full: full vector of nonlinear unknowns
        :param str_state_denoter: string which denotes the phase-state of the system
        :return vec_nonlin_unknws_full: (possibly) updated nonlinear unknowns and updates state denoter
        :return str_state_denoter: (possibly) updated string which denotes the phase-state of the system
        """
        # Do two-phase vapor-solid flash:
        # Reduced set of unknowns:
        #            4      5      7       8
        # --- X = [y_h2o, y_co2, nu_vap, nu_sol]        (PYTHON COUNTING!)
        # Map to :   0      1      2       3            --> size (4,)
        # Equations used: eq_0  eq_1  eq_7  eq_8
        # Map to:           0     1     2    3          --> size (4, 4)
        # Map full set of unknowns to reduced set:
        vec_nonlin_unknws_reduced = np.zeros((4,))
        vec_nonlin_unknws_reduced[0:2] = vec_nonlin_unknws_full[4:6]
        vec_nonlin_unknws_reduced[2:] = vec_nonlin_unknws_full[7:]

        # Compute initial residual equations:
        vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)
        dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknws_full)

        # Map full residual to reduced residual:
        vec_residual = np.zeros((4,))
        vec_residual[0:2] = dummy_residual[0:2]
        vec_residual[2:] = dummy_residual[7:]

        sca_iter_counter = 0
        temp_tolerance = self.sca_tolerance
        while (np.linalg.norm(vec_residual) > temp_tolerance) or (sca_iter_counter < 3):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            dummy_jacobian = self.compute_jacobian_full_system(vec_element_comp, vec_nonlin_unknws_full)

            # Map full jacobian to reduced jacobian, from (9, 9) to (4, 4):
            mat_jacobian = np.zeros((4, 4))
            mat_jacobian[0:2, 0:2] = dummy_jacobian[0:2, 4:6]
            mat_jacobian[0:2, 2:] = dummy_jacobian[0:2, 7:]
            mat_jacobian[2:, 0:2] = dummy_jacobian[7:, 4:6]
            mat_jacobian[2:, 2:] = dummy_jacobian[7:, 7:]

            # Solve linear system:
            vec_nonlin_update_reduced = -np.linalg.solve(mat_jacobian, vec_residual)

            # Update non-linear unknowns:
            vec_nonlin_unknws_reduced = vec_nonlin_unknws_reduced + vec_nonlin_update_reduced

            # Map back to full solution:
            vec_nonlin_unknws_full[4:6] = vec_nonlin_unknws_reduced[0:2]
            vec_nonlin_unknws_full[7:] = vec_nonlin_unknws_reduced[2:]

            # Recompute composition:
            vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)

            # Recompute residual:
            dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknws_full)

            # Map full residual to reduced residual:
            vec_residual = np.zeros((4,))
            vec_residual[0:2] = dummy_residual[0:2]
            vec_residual[2:] = dummy_residual[7:]

            # Increment iteration counter:
            sca_iter_counter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (sca_iter_counter == 20) or (sca_iter_counter == 40):
                temp_tolerance = temp_tolerance * 10

        # Check if now one phase fraction has changed to <= 0:
        # First check for negative values in the liquid and vapor component fractions as well as phase fractions:
        vec_nonlin_unknws_full, temp_index = self.eval_bounds_nonlin_unkwns(vec_nonlin_unknws_full)

        # Check phase denoter (if two-phase flash indicates single phase state):
        if temp_index[7]:
            # Vapor phase not present after two-phase flash, so single phase solid:
            str_state_denoter = '100'
        elif temp_index[8]:
            # Solid phase not present after two-phase flash, so single phase vapor:
            str_state_denoter = '010'
        return vec_nonlin_unknws_full, str_state_denoter

    def two_phase_liq_sol(self, vec_element_comp, vec_nonlin_unknws_full, str_state_denoter):
        """
        Class method which solves a two-phase flash for liquid-solid equilibrium
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknws_full: full vector of nonlinear unknowns
        :param str_state_denoter: string which denotes the phase-state of the system
        :return vec_nonlin_unknws_full: (possibly) updated nonlinear unknowns and updates state denoter
        :return str_state_denoter: (possibly) updated string which denotes the phase-state of the system
        """
        # Reduced set of unknowns:
        # -- X = [x_h2o, x_co2, x_co3, x_ca, nu_liq, nu_sol]    (PYTHON COUNTING!)
        # Map to:   0      1      2     3      6       8        --> size (6,)
        # Equations used: eq_0  eq_1  eq_2  eq_3  eq_6  eq_7
        # Map to:           0     1     2    3     4     5      --> size (6, 6)
        # Map full set of unknowns to reduced set:
        vec_nonlin_unknws_reduced = np.zeros((6,))
        vec_nonlin_unknws_reduced[0:4] = vec_nonlin_unknws_full[0:4]
        vec_nonlin_unknws_reduced[4] = vec_nonlin_unknws_full[6]
        vec_nonlin_unknws_reduced[5] = vec_nonlin_unknws_full[8]

        # Compute full residual:
        vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)
        dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknws_full)

        # Map full residual to reduced residual:
        vec_residual = np.zeros((6,))
        vec_residual[0:4] = dummy_residual[0:4]
        vec_residual[4] = dummy_residual[6]
        vec_residual[5] = dummy_residual[7]

        sca_iter_counter = 0
        temp_tolerance = self.sca_tolerance
        while (np.linalg.norm(vec_residual) > temp_tolerance) or (sca_iter_counter < 3):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            dummy_jacobian = self.compute_jacobian_full_system(vec_element_comp, vec_nonlin_unknws_full)

            # Map full jacobian to reduced jacobian, from (9, 9) to (6, 6):
            mat_jacobian = np.zeros((6, 6))
            mat_jacobian[0:4, 0:4] = dummy_jacobian[0:4, 0:4]
            mat_jacobian[0:4, 4] = dummy_jacobian[0:4, 6]
            mat_jacobian[0:4, 5] = dummy_jacobian[0:4, 8]
            mat_jacobian[4, 0:4] = dummy_jacobian[6, 0:4]
            mat_jacobian[4, 4] = dummy_jacobian[6, 6]
            mat_jacobian[4, 5] = dummy_jacobian[6, 8]
            mat_jacobian[5, 0:4] = dummy_jacobian[7, 0:4]
            mat_jacobian[5, 4] = dummy_jacobian[7, 6]
            mat_jacobian[5, 5] = dummy_jacobian[7, 8]

            # Solve linear system:
            vec_nonlin_update_reduced = -np.linalg.solve(mat_jacobian, vec_residual)

            # Update non-linear unknowns:
            vec_nonlin_unknws_reduced = vec_nonlin_unknws_reduced + vec_nonlin_update_reduced

            # Map back to full solution:
            vec_nonlin_unknws_full[0:4] = vec_nonlin_unknws_reduced[0:4]
            vec_nonlin_unknws_full[6] = vec_nonlin_unknws_reduced[4]
            vec_nonlin_unknws_full[8] = vec_nonlin_unknws_reduced[5]

            # Recompute composition:
            vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)

            # Recompute residual:
            dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp,
                                                               vec_nonlin_unknws_full)

            # Map full residual to reduced residual:
            vec_residual = np.zeros((6,))
            vec_residual[0:4] = dummy_residual[0:4]
            vec_residual[4] = dummy_residual[6]
            vec_residual[5] = dummy_residual[7]

            # Increment iteration counter:
            sca_iter_counter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (sca_iter_counter == 20) or (sca_iter_counter == 40):
                temp_tolerance = temp_tolerance * 10

        # Check if now one phase fraction has changed to <= 0:
        # First check for negative values in the liquid and vapor component fractions as well as phase fractions:
        vec_nonlin_unknws_full, temp_index = self.eval_bounds_nonlin_unkwns(vec_nonlin_unknws_full)

        # Check phase denoter (if two-phase flash indicates single phase state):
        if temp_index[6]:
            # Liquid phase not present after two-phase flash, so single phase solid:
            str_state_denoter = '100'
        elif temp_index[8]:
            # Solid phase not present after two-phase flash, so single phase liquid:
            str_state_denoter = '001'
        return vec_nonlin_unknws_full, str_state_denoter

    def two_phase_liq_vap(self, vec_element_comp, vec_nonlin_unknws_full, str_state_denoter):
        """
        Class method which solves a two-phase flash for liquid-vapor equilibrium
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknws_full: full vector of nonlinear unknowns
        :param str_state_denoter: string which denotes the phase-state of the system
        :return vec_nonlin_unknws_full: (possibly) updated nonlinear unknowns and updates state denoter
        :return str_state_denoter: (possibly) updated string which denotes the phase-state of the system
        """
        # Reduced set of unknowns:
        #            0      1      2       3    4      5      6       7
        # --- X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap]     (PYTHON COUNTING!)
        # Map to :   0      1      2       3    4      5      6       7         --> size (8,)
        # Equations used: eq_0  eq_1  eq_2  eq_3  eq_4  eq_5  eq_7  eq_8
        # Map to:           0     1     2    3     4     5      6     7         --> size (8, 8)
        # Map full set of unknowns to reduced set:
        vec_nonlin_unknws_reduced = vec_nonlin_unknws_full[:-1]

        # Compute full residual:
        vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)
        dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp, vec_nonlin_unknws_full)

        # Map full residual to reduced residual:
        vec_residual = np.zeros((8,))
        vec_residual[0:6] = dummy_residual[0:6]
        vec_residual[6:] = dummy_residual[7:]

        sca_iter_counter = 0
        temp_tolerance = self.sca_tolerance
        while (np.linalg.norm(vec_residual) > temp_tolerance) or (sca_iter_counter < 3):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            dummy_jacobian = self.compute_jacobian_full_system(vec_element_comp, vec_nonlin_unknws_full)

            # Map full jacobian to reduced jacobian, from (9, 9) to (6, 6):
            mat_jacobian = np.zeros((8, 8))
            mat_jacobian[0:6, 0:6] = dummy_jacobian[0:6, 0:6]
            mat_jacobian[0:6, 6:] = dummy_jacobian[0:6, 6:-1]
            mat_jacobian[6:, 0:6] = dummy_jacobian[7:, 0:6]
            mat_jacobian[6:, 6:] = dummy_jacobian[7:, 6:-1]

            # Solve linear system:
            vec_nonlin_update_reduced = -np.linalg.solve(mat_jacobian, vec_residual)

            # Update non-linear unknowns:
            vec_nonlin_unknws_reduced = vec_nonlin_unknws_reduced + vec_nonlin_update_reduced

            # Map back to full solution:
            vec_nonlin_unknws_full[:-1] = vec_nonlin_unknws_reduced

            # Recompute composition:
            vec_component_comp = self.eval_comp(vec_nonlin_unknws_full)

            # Recompute residual:
            dummy_residual = self.compute_residual_full_system(vec_element_comp, vec_component_comp,
                                                               vec_nonlin_unknws_full)

            # Map full residual to reduced residual:
            vec_residual = np.zeros((8,))
            vec_residual[0:6] = dummy_residual[0:6]
            vec_residual[6:] = dummy_residual[7:]

            # Increment iteration counter:
            sca_iter_counter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (sca_iter_counter == 20) or (sca_iter_counter == 40):
                temp_tolerance = temp_tolerance * 10

        # Check if now one phase fraction has changed to <= 0:
        # First check for negative values in the liquid and vapor component fractions as well as phase fractions:
        vec_nonlin_unknws_full, temp_index = self.eval_bounds_nonlin_unkwns(vec_nonlin_unknws_full)

        # Check phase denoter (if two-phase flash indicates single phase state):
        if temp_index[6]:
            # Liquid phase not present after two-phase flash, so single phase vapor:
            str_state_denoter = '010'
        elif temp_index[7]:
            # Vapor phase not present after two-phase flash, so single phase liquid:
            str_state_denoter = '001'
        return vec_nonlin_unknws_full, str_state_denoter

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
        self.vec_liquid_molefrac = np.append(vec_nonlin_unknowns[:4], [0], axis=0)
        self.vec_vapor_molefrac = np.append(vec_nonlin_unknowns[4:6], [0, 0, 0], axis=0)
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
            # If any immobile phase present, set mobile_phase saturation to 0:
            vec_saturation_mobile_phases = np.zeros((2,))
            # raise Exception('WARNING: NO MOBILE PHASE PRESENT!!!')
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
                                              self.vec_liquid_molefrac * vec_fractional_flow_phases[1] * \
                                              self.vec_actual_density[1]
        return 0

    def eval_if_two_phas(self, vec_element_comp, vec_nonlin_unknowns, str_state_denoter):
        """
        Class method which solves a two-phase flash for problem depending on phase-state of the system
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknowns: full vector of nonlinear unknowns
        :param str_state_denoter: string which denotes the phase-state of the system
        :return vec_nonlin_unknowns: (possibly) updated nonlinear unknowns and updates state denoter
        :return str_state_denoter: (possibly) updated string which denotes the phase-state of the system
        """
        if str_state_denoter == '110':
            # Do two-phase vapor-solid flash:
            vec_nonlin_unknowns, str_state_denoter = self.two_phase_vap_sol(vec_element_comp,
                                                                            vec_nonlin_unknowns,
                                                                            str_state_denoter)
        elif str_state_denoter == '101':
            # Do two-phase liquid-solid flash:
            vec_nonlin_unknowns, str_state_denoter = self.two_phase_liq_sol(vec_element_comp,
                                                                            vec_nonlin_unknowns,
                                                                            str_state_denoter)
        elif str_state_denoter == '011':
            # Do two-phase liquid-vapor flash:
            vec_nonlin_unknowns, str_state_denoter = self.two_phase_liq_vap(vec_element_comp,
                                                                            vec_nonlin_unknowns,
                                                                            str_state_denoter)
        return vec_nonlin_unknowns, str_state_denoter

    def eval_if_one_phase(self, vec_element_comp, vec_nonlin_unknowns, str_state_denoter):
        """
        Class methods which assigns based on current state denoter elemental compositions as solution to the nonlinear
        problem in case of single phase system (in this case no reaction amongst components or elements in the same
        phase, therefore simply assign element composition to vector of nonlinear unknowns)
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknowns: full vector of nonlinear unknowns
        :param str_state_denoter: string which denotes the phase-state of the system
        :return vec_nonlin_unknowns: (possibly) updated nonlinear unknowns and updates state denoter
        :return str_state_denoter: (possibly) updated string which denotes the phase-state of the system
        """
        if str_state_denoter == '100':
            # No liquid and vapor phase present, therefore in state 100:
            print('Only solid phase present, check element composition, must be an error somewhere!!!\n')
        elif str_state_denoter == '001':
            # Single phase liquid, all elements are soluble in water so ze == zc(0:-1):
            vec_nonlin_unknowns[0:4] = vec_element_comp[0:4]
        elif str_state_denoter == '010':
            # Single phase gas, only first two elements appear in gas phase ze == zc(0:-1):
            vec_nonlin_unknowns[4:6] = vec_element_comp[0:2]
        return vec_nonlin_unknowns, str_state_denoter

    def calc_thermodynamic_state(self, state):
        """
        Class method which computes the thermodynamic state of the system based on the current state
        :param state: vector with state related parameters [pressure, element_comp_0, ..., element_comp_N-1]
        :return: save converged vector of non-linear unknowns to self
        """
        # For state at current iteration, the piece of code will do the following steps:
        #   1) Initialize unknowns (to be solving in physical region)
        #   2) Compute three-phase flash
        #   3) Determine if in 3-phase or not
        #       3.1) If 3-phase proceeed to 4
        #       3.2) If not 3-phase, compute 2-phase flash
        # -----------
        # STEP 1:
        # -----------
        # Element composition vector:
        vec_state_as_np = np.asarray(state)
        vec_element_composition = np.append(vec_state_as_np[1:],
                                            [(1 - np.sum(state[1:])) / 2, (1 - np.sum(state[1:])) / 2])
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        vec_element_composition = self.out_bound_composition(vec_element_composition)

        # Find new K_water and K_co2, the thermodynamic constants:
        # sca_k_water = interp1d(self.vec_pressure_range_k_values, self.vec_thermo_equi_const_k_water)(sca_pressure)
        # sca_k_co2 = interp1d(self.vec_pressure_range_k_values, self.vec_thermo_equi_const_k_co2)(sca_pressure)
        sca_k_water = 0.1080
        sca_k_co2 = 1149
        self.vec_k_values = np.array([sca_k_water, sca_k_co2]).flatten()

        # Get more educated initial guess for three-phase flash:
        two_phase_liq_molefrac, two_phase_vap_molefrac = self.init_three_phase_flash()

        # Initialize vector of nonlinear unknowns used for the three phase flash:
        vec_nonlin_unknowns = self.construct_init_nonlinear_unknowns(two_phase_liq_molefrac, two_phase_vap_molefrac)

        # -----------
        # STEP 2:
        # -----------
        # Compute three-phase flash:
        # Create vector with unknowns (initialized):
        # NOTE: vec_x = [x_h20, x_co2, x_ca+2, x_co3, x_caco3], similar ordering for z_c and y, etc.
        vec_nonlin_unknowns, temp_index = self.three_phase_flash(vec_element_composition, vec_nonlin_unknowns)

        # Determine if system is in three-phase or less:
        # Phase denotation system according to AD-GPRS logic for three-phase system:
        # Water = 0, Oil = 1, Gas = 2 || Liquid = 0, Vapor = 1, Solid = 2
        # system_state 000 001 010 100 011 101 110 111
        # phase_0       -   x   -   -   x   x   -   x
        # phase_1       -   -   x   -   x   -   x   x
        # phase_2       -   -   -   x   -   x   x   x
        # Since zero phases present is not possible in our system, there are 2^{n}-1 states possible!
        str_state_denoter = self.state_denoter(temp_index)

        # -----------------------------------------------------------------------------------
        # STEP 3.1 & 3.2: Compute two-phase flash if necessary, else proceed to STEP 4
        # -----------------------------------------------------------------------------------
        # Calculate new phase fractions if in (possible) two-phase region:
        vec_nonlin_unknowns, str_state_denoter = self.eval_if_two_phas(vec_element_composition,
                                                                       vec_nonlin_unknowns,
                                                                       str_state_denoter)

        # Evaluate if in single-phase after possible two-phase flash:
        vec_nonlin_unknowns, str_state_denoter = self.eval_if_one_phase(vec_element_composition,
                                                                        vec_nonlin_unknowns,
                                                                        str_state_denoter)

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
        # For state at current iteration, the piece of code will do the following steps:
        #   1) Initialize unknowns (to be solving in physical region)
        #   2) Compute three-phase flash
        #   3) Determine if in 3-phase or not
        #       3.1) If 3-phase proceeed to 4
        #       3.2) If not 3-phase, compute 2-phase flash
        #   4) Based on particular phase-state (7 total), compute all state related variables
        #       such as saturation, relative permeability, etc.
        #   5) Compute operator values
        # print(state)
        # Extract state (pres, comp) and normalize elemental composition:
        vec_state_as_np = np.asarray(state)
        vec_element_composition = np.append(vec_state_as_np[1:],
                                            [(1 - np.sum(state[1:])) / 2, (1 - np.sum(state[1:])) / 2])
        vec_elem_comp_copy = np.array(vec_element_composition, copy=True)
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        vec_element_composition = self.out_bound_composition(vec_element_composition)

        # -----------
        # STEP 1, 2, 3:
        # -----------
        self.calc_thermodynamic_state(state)

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
        sca_total_element_density = np.sum(np.dot(self.mat_rate_annihilation, self.vec_composition)) / \
                                    self.sca_phase_frac_weighted_by_density
        sca_rock_compres_factor = 1 + self.sca_compressibility_solid * (sca_pressure - self.sca_ref_pres)

        # Alpha operator:
        values[0] = sca_rock_compres_factor * sca_total_element_density * vec_elem_comp_copy[0]
        values[1] = sca_rock_compres_factor * sca_total_element_density * vec_elem_comp_copy[1]
        values[2] = sca_rock_compres_factor * sca_total_element_density * np.sum(vec_elem_comp_copy[2:])

        # Beta operator:
        temp_beta_operator = self.sca_trans_multiplier * \
                             np.dot(self.mat_rate_annihilation, (self.sca_total_mobility *
                                                                 self.vec_fractional_flow_components))
        values[3] = temp_beta_operator[0]
        values[4] = temp_beta_operator[1]
        values[5] = np.sum(temp_beta_operator[2:])

        if np.sum(state[1:]) < 0.01:
            print('\n------------------------ERROR------------------------')
            print('\t\t\tNO FLUID PHASE FOUND!!!')
            print('------------------------ERROR------------------------\n')
            # raise Exception('NO FLUID FOUND')
        elif any(np.isnan(values)):
            print('\n------------------------ERROR------------------------')
            print('\t\t\tNAN Returned for values!!!')
            print('------------------------ERROR------------------------\n')
            # raise Exception('NO FLUID FOUND')

        if self.bool_debug_mode_on:
            print('For P = {:f}, z_e = [{:f},{:f}]: Alpha = [{:f},{:f},{:f}] and Beta = [{:f},{:f},{:f}]'.
                  format(state[0], state[1], state[2], values[0], values[1], values[2], values[3], values[4], values[5]))

            print('z_c = {:s}, F_c = {:s}, \n\t lam_T = {:f}, rho_T_E = {:f}, a = {:f}'.
                  format(np.array_str(self.vec_composition),
                         np.array_str(self.vec_fractional_flow_components),
                         self.sca_total_mobility,
                         sca_total_element_density, sca_rock_compres_factor))

        if self.output_operators:
            state_return = np.array([sca_pressure, vec_element_composition[0], vec_element_composition[1]])
            return state_return, values.flatten()
        return 0

    def eval_state_elem(self, state):
        """
        Class methods which evaluates the properties of the particular state of the system:
        :param state: state variables [pres, elem_comp_0, ..., elem_comp_N-1]
        :return: returns the phase split (x,y,w) and phase saturations
        """
        # Extract state (pres, comp) and normalize elemental composition:
        vec_state_as_np = np.asarray(state)
        vec_element_composition = np.append(vec_state_as_np[1:],
                                            [(1 - np.sum(state[1:])) / 2, (1 - np.sum(state[1:])) / 2])
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        vec_element_composition = self.out_bound_composition(vec_element_composition)

        # -----------
        # STEP 1, 2, 3:
        # -----------
        self.calc_thermodynamic_state(state)

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
        return 0


class element_acc_flux_data():
    """
    This class holds all the necessary (user)input data for evaluating the accumulation, flux, and other properties
    for the elements based physics
    """
    def __init__(self, mat_rate_annihilation, vec_pressure_range_k_values, vec_thermo_equi_const_k_water,
                                  vec_thermo_equi_const_k_co2, sca_k_caco3, sca_tolerance,
                                  sca_ref_pres, sca_density_water_stc, sca_compressibility_water, sca_density_gas_stc,
                                  sca_compressibility_gas, sca_density_solid_stc, sca_compressibility_solid,
                                  vec_res_sat_mobile_phases, vec_brooks_corey_exp, vec_end_point_rel_perm,
                                  vec_viscosity_mobile_phases, sca_transmissibility_exp, min_comp):
        # Assign data to data structure:
        self.mat_rate_annihilation = mat_rate_annihilation
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
        self.min_comp = min_comp


class chemical_rate_evaluator(element_acc_flux_etor):
    # Simplest class existing to mankind:
    def __init__(self, elements_data, bool_trans_upd=True, bool_debug_mode_on=False, output_operators=False):
        # Initialize base-class
        super().__init__(elements_data, bool_trans_upd, bool_debug_mode_on, output_operators)

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
            # raise Exception('WARNING: NO MOBILE PHASE PRESENT!!!')
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
        # Extract state (pres, comp) and normalize elemental composition:
        vec_state_as_np = np.asarray(state)
        vec_element_composition = np.append(vec_state_as_np[1:],
                                            [(1 - np.sum(state[1:])) / 2, (1 - np.sum(state[1:])) / 2])
        sca_pressure = vec_state_as_np[0]

        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        vec_element_composition = self.out_bound_composition(vec_element_composition)

        # -----------
        # STEP 1, 2, 3:
        # -----------
        self.calc_thermodynamic_state(state)

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
