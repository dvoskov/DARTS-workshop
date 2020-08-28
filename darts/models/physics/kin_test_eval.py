import numpy as np
from darts.engines import *
from darts.physics import *
import os.path as osp
from scipy.interpolate import interp1d

physics_name = osp.splitext(osp.basename(__file__))[0]


# Define our own operator evaluator class
class component_acc_flux_etor(operator_set_evaluator_iface):
    def __init__(self, kin_rate, order_react, equi_prod, log_based, num_comp, min_z):
        super().__init__()  # Initialize base-class
        # Extra properties for kinetics:
        self.kin_rate = kin_data.kin_rate
        self.order_react = kin_data.order_react
        self.equi_prod = kin_data.equi_prod
        self.log_based = log_based
        self.num_comp = num_comp
        self.min_z = min_z

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

        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z);

        # Alpha operator:
        num_alpha_op = self.num_comp
        for ith_alpha in range(num_alpha_op):
            values[ith_alpha] = vec_composition[ith_alpha]

        # Beta operator:
        num_beta_op = self.num_comp
        for ith_beta in range(num_beta_op):
            values[ith_beta + num_alpha_op] = vec_composition[ith_beta]

        # Set "mobility" of solid phase to zero:
        values[1 + num_alpha_op] = 1e-15 * 1

        #TODO: UPDATE PARAMETERS AND TAKE OUTSIDE OF EVALUATOR AND INTO DATA STRUCTURE
        num_kin_op = self.num_comp
        stoich_matrix = np.array([1, -1, 0])

        # NEW WAY:
        for ith_gamma in range(num_kin_op):
            # Store each kinetic operator:
            values[ith_gamma + num_alpha_op + num_beta_op] = -self.kin_rate * (1 - (vec_composition[0]/self.equi_prod)**self.order_react) * \
                                                             stoich_matrix[ith_gamma] * (vec_composition[1] - self.min_z)

        # Store porosity operator:
        num_poro_op = 1
        values[num_kin_op + num_alpha_op + num_beta_op] = 1 - vec_composition[1]
        return 0


class chemical_rate_evaluator(operator_set_evaluator_iface):
    # Simplest class existing to mankind:
    def __init__(self):
        # Initialize base-class
        super().__init__()

    def evaluate(self, state, values):
        # Easiest example, constant volumetric phase rate:
        values[0] = 0   # vapor phase
        values[1] = 1   # liquid phase

        # Usually some steps will be executed to estimate unit volumetric flow rate based on current state (when
        # multiplied with a pressure difference one obtains actual volumetric flow rate)
        return 0
