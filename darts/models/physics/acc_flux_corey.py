from darts.engines import  operator_set_evaluator_iface


# Accumulation is taken from parent acc_flux
# Flux is calculated based on Corey approximation calculated for saturation from water_sat_ev


class opt_acc_flux_corey(operator_set_evaluator_iface):
    def __init__(self, acc_flux, water_sat_ev):
        super().__init__()
        #use this to generate accumulation operator
        self.acc_flux = acc_flux
        self.water_sat_ev = water_sat_ev
        self.corey_params = [2, 6, 0.0, 0.0, 2655.25, 1542.33]
        self.define_corey_params(self.corey_params[0], self.corey_params[1], self.corey_params[2],
                                 self.corey_params[3], self.corey_params[4], self.corey_params[5])

    def define_corey_params (self, e1, e2, sr1, sr2, fmax1, fmax2):
        self.e1 = e1
        self.e2 = e2
        self.sr1 = sr1
        self.sr2 = sr2
        self.fmax1 = fmax1
        self.fmax2 = fmax2
        self.corey_params = [e1, e2, sr1, sr2, fmax1, fmax2]

    def evaluate(self, state, values):
         self.acc_flux.evaluate(state, values)
         water_sat = self.water_sat_ev.evaluate(state)
         eps = 1e-10

         water_eff_sat = (water_sat - self.sr1) / (1 - self.sr1 - self.sr2) 
         water_eff_sat = min (water_eff_sat, 1.0)
         water_eff_sat = max (water_eff_sat, 0.0)
         f1 = self.fmax1 * (water_eff_sat + eps) ** (self.e1)
         f2 = self.fmax2 * (1 - water_eff_sat + eps) ** (self.e2)
         values[2] = f1
         values[3] = f2
         #print (self.e1, self.e2)
         #print (state, values)
         return 0
