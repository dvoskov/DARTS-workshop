import numpy as np
import h5py

def get_rates(rates_filename):
    a = np.genfromtxt(rates_filename, names=True)
    b = np.delete(a, 0, 0)
    return b

def get_h5_vars (vars_filename):
    f = h5py.File(vars_filename, 'r')
    P_T_Zc_Sp = f.get('FLOW_TRANSPORT/PTZS') # sometimes this is different
    P = P_T_Zc_Sp[:,0,:]
    T = P_T_Zc_Sp[:,1,:]
    ZS = P_T_Zc_Sp[:,2:-1,:]
    f.close()
    return P, T, ZS

def save_vars_keywords (folder, mesh):
    #assert isinstance(mesh, conn_mesh)
    mesh.save_pressure(folder + r"\pressure.in")
    mesh.save_poro(folder + r"\poro.in")
    mesh.save_volume(folder + r"\volume.in")
    mesh.save_zmf(folder + r"\zmf.in")
    mesh.save_temperature(folder + r"\temperature.in")

