from darts.engines import conn_mesh, ms_well, ms_well_vector
import numpy as np
from math import pi


class Reservoir:
    def __init__(self):
        # create mesh object
        self.mesh = conn_mesh()

        # reservoir is supposed to be discretized by external tool
        # read connections and transmissibilities from file
        self.mesh.init('conn2p_2D.txt')
        self.nb = self.mesh.n_blocks

        self.nx = 60
        self.ny = 40
        self.nz = 1

        # Create numpy arrays wrapped around mesh data (no copying)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.porosity = np.array(self.mesh.poro, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.cond = np.array(self.mesh.rock_cond, copy=False)        

        # Set uniform value for all elements of volume array
        # 30m x 30m x 2.5m = 2250 m3
        self.volume.fill(2250)
        self.volume[0:self.nb:60] = 8e9
        self.volume[59:self.nb:60] = 8e9

        # Load heterogeneous porosity values from file,
        # make the resulting ndarray flat,
        # and assign it`s values to the existing array ([:] is very important!)
        self.porosity[:] = np.genfromtxt('poro_2D.txt', skip_header=True, skip_footer=True).flatten()
        
        # Constant definitions
        self.depth.fill(2500)
        self.cond.fill(200)
        self.hcap.fill(2200)        

        self.wells = []

    def add_well(self, name):
        well = ms_well()
        well.name = name
        # assume length equal to reservoir layer height (2.5m)
        # assume wellbore diameter equal to 15 cm
        well.segment_volume = 2.5 * pi * 0.15**2 / 4
        well.well_head_depth = 2502.5 + 2.5 / 2
        well.well_body_depth = 2502.5 + 2.5 / 2
        well.segment_transmissibility = 1e3
        well.segment_depth_increment = 2.5
        self.wells.append(well)

    def add_perforation(self, well, i, j, k, well_index):
        # switch to 0-based index
        i -= 1
        j -= 1
        k -= 1
        # compute reservoir block index
        res_block = k * self.nx * self.ny + j * self.nx + i
        # assume 1 segment per layer
        # set well segment index (well block) equal to index of perforation layer
        well_block = k
        # add completion
        well.perforations = well.perforations + [(well_block, res_block, well_index)]

    def init_wells(self):

        # take well index values from external mesh discretizer (ADGPRS)
        self.add_well("I1")
        self.add_perforation(well=self.wells[-1], i=15, j=20, k=1, well_index=10)

        self.add_well("P1")
        self.add_perforation(self.wells[-1], 48, 20, 1, 10)

        self.mesh.add_wells(ms_well_vector(self.wells))
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        