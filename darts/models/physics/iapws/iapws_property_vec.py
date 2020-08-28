import numpy as np
from math import sqrt, log, exp


def _Backward1_T_Ph_vec(P, h):
    """
    Backward equation in vector form for region 1, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure [MPa]
    h : float
        Specific enthalpy [kJ/kg]

    Returns
    -------
    T : float
        Temperature [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 11

    """
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6]
    J = [0, 1, 2, 6, 22, 32, 0, 1, 2, 3, 4, 10, 32, 10, 32, 10, 32, 32, 32, 32]
    n = [-0.23872489924521e3, 0.40421188637945e3, 0.11349746881718e3,
         -0.58457616048039e1, -0.15285482413140e-3, -0.10866707695377e-5,
         -0.13391744872602e2, 0.43211039183559e2, -0.54010067170506e2,
         0.30535892203916e2, -0.65964749423638e1, 0.93965400878363e-2,
         0.11573647505340e-6, -0.25858641282073e-4, -0.40644363084799e-8,
         0.66456186191635e-7, 0.80670734103027e-10, -0.93477771213947e-12,
         0.58265442020601e-14, -0.15020185953503e-16]

    Pr = P/1
    nu = h/2500
    T = np.zeros(P.shape)
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (nu+1)**j
    return T