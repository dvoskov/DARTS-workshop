from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector, operator_set_evaluator_iface
import numpy as np

from darts.models.physics_sup.property_container import *
from darts.models.physics_sup.properties_basic import *
from darts.models.physics_sup.operator_evaluator_sup import *
from darts.models.physics_sup.physics_comp_sup import Compositional

import matplotlib.pyplot as plt

''' Map for 2D model '''
def create_map(lx, ly, nx, ny):

    map = np.ones((nx, ny))

    x1 = 120
    y1 = 80

    DX = lx / nx
    DY = ly / ny

    nx1 = int(np.ceil(x1 / DX))
    nx2 = int(np.floor((lx - x1) / DX))
    ny1 = int(np.ceil(y1 / DY))
    ny2 = int(np.floor((ly - y1) / DY))

    map[nx1:nx2, ny1:ny2] = 0

    map = np.reshape(map, (nx * ny,), order='F')

    return map

''' Model class creation '''
class Model(DartsModel):
    def __init__(self, grid_1D=True):    
        # Call base class constructor
        super().__init__()
        self.grid_1D = grid_1D

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """ Basic parameters """
        self.zero = 1e-12
        init_ions = 0.5
        solid_init = 0.7
        equi_prod = (init_ions / 2) ** 2
        solid_inject = self.zero
        trans_exp = 3
        self.combined_ions = True
        self.init_pres = 95
        self.physics_type = 'kin'  # equi or kin

        """ Reservoir """
        if grid_1D:
            self.dx = 1
            self.dy = 1
            self.dz = 1
            perm = 100 / (1 - solid_init) ** trans_exp
            (self.nx, self.ny, self.nz) = (1000, 1, 1)
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy, dz=self.dz,
                                             permx=perm, permy=perm, permz=perm/10, poro=1, depth=1000)

            """ Well locations """
            self.reservoir.add_well("INJ_GAS")
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

            self.reservoir.add_well("PROD")
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=1, k=1, multi_segment=False)

            self.inj_gas_rate = 0.2

            zc_fl_init = [self.zero / (1 - solid_init), init_ions]
            zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
            self.ini_comp = [x * (1 - solid_init) for x in zc_fl_init]

            self.map = []
        else:
            (Lx, Lz) = (600, 240)
            (self.nx, self.ny, self.nz) = (60, 10, 24)
            self.dx = Lx / self.nx
            self.dz = Lz / self.nz

            self.map = create_map(Lx, Lz, self.nx, self.nz)

            perm = np.ones(self.nx * self.nz) * 100 / (1 - solid_init) ** trans_exp

            # Add inclination in y-direction:
            self.depth = np.ones((self.nx * self.nz,)) * 1000
            for j in range(self.nz):
                self.depth[j * self.nx:(j + 1) * self.nx] += j * self.dz

            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=1, nz=self.nz, dx=self.dx, dy=self.ny, dz=self.dz,
                                             permx=perm, permy=perm, permz=perm, poro=1, depth=self.depth)
            
            """ Well locations """
            self.reservoir.add_well("PROD_" + str(1))
            for ii in range(self.nz):
                # self.reservoir.add_well("PROD_" + str(ii) + str(1))
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=1, k=ii + 1,
                                               multi_segment=False)

            self.inj_gas_rate = 1000 / self.nz * 2
            self.inj_wat_rate = 200 / self.nz * 2

            solid_void = 0.2
            if self.combined_ions:
                zc_fl_init = [self.zero / (1 - solid_init), init_ions]
            else:
                # zc_fl_init = [self.zero / (1 - solid_init), init_ions, self.zero / (1 - solid_init)]
                zc_fl_init = [self.zero / (1 - solid_init), init_ions / 2, init_ions / 2]
            zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
            self.ini_comp = [x * (1 - solid_init) for x in zc_fl_init]
            self.ini_void = [x * (1 - solid_void) for x in zc_fl_init]

        """ Property container definition """
        # Create property containers:
        if self.combined_ions:
            components_name = ['CO2', 'Ions', 'H2O', 'CaCO3']
            Mw = [44.01, (40.078 + 60.008) / 2, 18.015, 100.086]
        else:
            components_name = ['CO2', 'Ca', 'CO3', 'H2O', 'CaCO3']
            Mw = [44.01, 40.078, 60.008, 18.015, 100.086]
            # Mw = [44.01, (40.078 + 60.008) / 2, (40.078 + 60.008) / 2, 18.015, 100.086]

        self.thermal = 0
        self.property_container = model_properties(phases_name=['gas', 'wat'],
                                                   components_name=components_name, rock_comp=1e-7, Mw=Mw,
                                                   min_z=self.zero / 10, diff_coef=1e-9 * 60 * 60 * 24,
                                                   solid_dens=[2000])

        self.components = self.property_container.components_name
        self.phases = self.property_container.phases_name

        """ Property correlations """
        if self.combined_ions:
            self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-1], self.zero)
        else:
            self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-12, 1e-1], self.zero)

        self.property_container.density_ev = dict([('gas', Density(compr=1e-4, dens0=100)),
                                                   ('wat', Density(compr=1e-6, dens0=1000))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.1)),
                                                     ('wat', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("wat"))])


        ne = self.property_container.nc + self.thermal
        self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e+0, ne, self.combined_ions)

        """ Activate physics """
        delta_volume = self.dx * self.dz * 10
        num_well_blocks = int(self.nz / 2)
        self.physics = CustomPhysics(self.property_container, self.timer, n_points=401, min_p=1, max_p=1000,
                                     min_z=self.zero/10, max_z=1-self.zero/10, cache=0, volume=delta_volume,
                                     num_wells=num_well_blocks)

        """ Boundary conditions """
        if self.combined_ions:
            zc_fl_inj_stream_gas = [1 - 2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
            zc_fl_inj_stream_liq = [2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
        else:
            zc_fl_inj_stream_gas = [1 - 3 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject), self.zero
                                    / (1 - solid_inject)]
            zc_fl_inj_stream_liq = [3 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject),
                                    self.zero / (1 - solid_inject)]

        zc_fl_inj_stream_gas = zc_fl_inj_stream_gas + [1 - sum(zc_fl_inj_stream_gas)]
        self.inj_stream_gas = [x * (1 - solid_inject) for x in zc_fl_inj_stream_gas]

        zc_fl_inj_stream_liq = zc_fl_inj_stream_liq + [1 - sum(zc_fl_inj_stream_liq)]
        self.inj_stream_wat = [x * (1 - solid_inject) for x in zc_fl_inj_stream_liq]

        """ Simulation parameters """
        self.params.trans_mult_exp = trans_exp
        self.params.first_ts = 0.001
        self.params.max_ts = 0.1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop

        self.runtime = 50
        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ Initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.init_pres, self.ini_comp)

        if len(self.map) > 0:
            nc = self.property_container.nc
            nb = self.reservoir.nb
            composition = np.array(self.reservoir.mesh.composition, copy=False)
            zc = np.zeros(nb)
            for i in range(nc-1):
                zc[:] = self.ini_comp[i]
                zc[self.map == 0] = self.ini_void[i]
                composition[i:(nc-1)*nb:nc-1] = zc
        return

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ_GAS" in w.name:
                w.control = self.physics.new_rate_inj(self.inj_gas_rate, self.inj_stream_gas, 0)
                # w.control = self.physics.new_bhp_inj(125, self.inj_stream_gas)
            elif "INJ_WAT" in w.name:
                w.control = self.physics.new_rate_inj(self.inj_wat_rate, self.inj_stream_wat, 1)
                # w.control = self.physics.new_bhp_inj(125, self.inj_stream_wat)
            else:
                w.control = self.physics.new_bhp_prod(95)


    def set_op_list(self):
        """ Custom operators for regions """
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1

        if self.grid_1D:
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
        else:
            self.slice_liq_inj = np.arange(0, np.int(self.nx * self.nz / 2) - 1, self.nx, dtype=int)
            self.slice_gas_inj = np.arange(np.int(self.nx * self.nz / 2), self.nx * self.nz - 1, self.nx, dtype=int)

            self.op_num[self.slice_gas_inj] = 2
            self.op_num[self.slice_liq_inj] = 3

            # self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor, self.physics.acc_flux_itor[1],
                            self.physics.acc_flux_itor[2]]

    """ Plot 1D model results """
    def print_and_plot_1D(self):
        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))

        rel_perm = np.zeros((self.reservoir.nb, 2))
        visc = np.zeros((self.reservoir.nb, 2))
        density = np.zeros((self.reservoir.nb, 3))
        density_m = np.zeros((self.reservoir.nb, 3))

        Xn = np.array(self.physics.engine.X, copy=True)

        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            rel_perm[ii, :] = kr
            visc[ii, :] = mu
            density[ii, :2] = rho
            density_m[ii, :2] = rho_m

            density[2] = self.property_container.solid_dens[-1]

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        phi = 1 - z_caco3
        # Start plot
        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 8,
                           }

        fig, ax = plt.subplots(3, 2, figsize=(8, 5), dpi=200, facecolor='w', edgecolor='k')
        names = ['z_co2', 'z_h2o', 'z_inert', 'P', 'Sg', 'phi']
        titles = ['$z_{CO_2}$ [-]', '$z_{H_2O}$ [-]', '$z_{w, Ca} + z_{w, CO_3}$ [-]',
                  '$P$ [bars]', '$s_g$ [-]', '$\phi$ [-]']
        for i in range(3):
            for j in range(2):
                n = i + j * 3
                vec = eval(names[n])
                im = ax[i, j].plot(vec)
                ax[i, j].set_title(titles[n], fontdict=font_dict_title)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.25, hspace=0.25)

        plt.tight_layout()
        plt.show()
    
    """ Plot 2D model results """
    def print_and_plot_2D(self):
        import matplotlib.pyplot as plt

        if self.combined_ions:
            plot_labels = ['$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', '$x_{w, Ca} + x_{w, CO_3}$ [-]']
        else:
            plot_labels = ['$z_{w, Ca+2}$ [-]', '$x_{w, Ca}$ [-]']

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 8,
                           }

        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))
        Xn = np.array(self.physics.engine.X, copy=True)

        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (
                    Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii * nc:(ii + 1) * nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        phi = 1 - z_caco3

        fig, ax = plt.subplots(3, 2, figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.set_cmap('jet')
        names = ['z_co2', 'z_h2o', 'z_inert', 'P', 'Sg', 'phi']
        titles = ['$z_{CO_2}$ [-]', '$z_{H_2O}$ [-]', '$z_{w, Ca} + z_{w, CO_3}$ [-]',
                  '$P$ [bars]', '$s_g$ [-]', '$\phi$ [-]']
        for i in range(3):
            for j in range(2):
                n = i + j * 3
                vec = eval(names[n])
                im = ax[i, j].imshow(vec.reshape(self.nz, self.nx))
                ax[i, j].set_title(titles[n], fontdict=font_dict_title)
                plt.colorbar(im, ax=ax[i, j])

        plt.tight_layout()
        plt.show()

""" Inherit default property container and redefine flash """
class model_properties(property_container):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11,
                 diff_coef=0.0, rock_comp=1e-6, solid_dens=None):
        # Call base class constructor
        # Cm = 0
        # super().__init__(phases_name, components_name, Mw, Cm, min_z, diff_coef, rock_comp, solid_dens)
        if solid_dens is None:
            solid_dens = []
        super().__init__(phases_name, components_name, Mw, min_z=min_z, diff_coef=diff_coef,
                         rock_comp=rock_comp, solid_dens=solid_dens)

    def run_flash(self, pressure, zc):

        nc_fl = self.nc - self.nm
        norm = 1 - np.sum(zc[nc_fl:])

        zc_r = zc[:nc_fl] / norm
        (xr, nu) = self.flash_ev.evaluate(pressure, zc_r)
        V = nu[0]

        if V <= 0:
            V = 0
            xr[1] = zc_r
            ph = [1]
        elif V >= 1:
            V = 1
            xr[0] = zc_r
            ph = [0]
        else:
            ph = [0, 1]

        for i in range(self.nc - 1):
            for j in range(2):
                self.x[j][i] = xr[j][i]

        self.nu[0] = V
        self.nu[1] = (1 - V)

        return ph

""" Define property evaluator """
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

        #  some arrays will be reused in thermal
        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

        nph = self.property.nph
        for i in range(nph):
            values[i + 0 * nph] = sat[i]
            values[i + 1 * nph] = rho[i]
            values[i + 2 * nph] = rho[i]
            values[i + 3 * nph] = kr[i]

        return 0

""" Inherit default physics and redefine operator set """
class CustomPhysics(Compositional):
    def __init__(self, property_container, timer, n_points, min_p, max_p, min_z, max_z, min_t=-1, max_t=-1, thermal=0,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=False,
                 volume=0, num_wells=0):

        self.delta_volume = volume
        self.num_well_blocks = num_wells

        super().__init__(property_container, timer, n_points, min_p, max_p, min_z, max_z, min_t, max_t, thermal,
                 platform, itor_type, itor_mode, itor_precision, cache)



    def set_operators(self, property_container, thermal):  # default definition of operators

        operators = self.operators_storage()

        operators.reservoir_operators[0] = ReservoirOperators(property_container)
        operators.wellbore_operators = ReservoirOperators(property_container)

        operators.reservoir_operators[1] = ReservoirWithSourceOperators(property_container, comp_inj_id=0,
                                                                        delta_volume=self.delta_volume,
                                                                        num_well_blocks=self.num_well_blocks)

        operators.reservoir_operators[2] = ReservoirWithSourceOperators(property_container, comp_inj_id=1,
                                                                        delta_volume=self.delta_volume,
                                                                        num_well_blocks=self.num_well_blocks)



        operators.rate_operators = RateOperators(property_container)

        operators.property_operators = DefaultPropertyEvaluator(property_container)

        return operators

""" Inherit default reservoir operators and add rate """
class ReservoirWithSourceOperators(ReservoirOperators):
    def __init__(self, property_container, comp_inj_id, thermal=0,
                 delta_volume=1000, num_well_blocks=12):
        super().__init__(property_container, thermal=thermal)  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        self.comp_inj_id = comp_inj_id
        self.delta_volume = delta_volume
        self.num_well_blocks = num_well_blocks

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        super().evaluate(state, values)
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        """ Delta operator for reaction """
        shift = nph * ne + nph + ne + ne * nph

        # mass flux injection (if comp 0 then pure CO2 in gas vorm if 2 then pure H2O in liquid)
        if self.comp_inj_id == 0:
            values[shift + 0] -= 1000 / self.num_well_blocks / self.delta_volume \
                               * self.property.density_ev['gas'].evaluate(pressure, 0) / 44.01
        elif self.comp_inj_id == 1:
            values[shift + 2] -= 200 / self.num_well_blocks / self.delta_volume \
                               * self.property.density_ev['wat'].evaluate(pressure, 0) / 18.015

        return 0
