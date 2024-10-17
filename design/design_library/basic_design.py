import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from design_library.base import *
from design_library.rational_opt import rational_opt

#############################################################

def design_converter_with_m(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, m_, fmin_, printing=False, plot=False):
    
    Vout_max_d = (Vout_ + Vd_max)
    Vout_max_ = Vout_max_d * 1.10
    Vin_min_ = Vin_min * 0.95
    fmin__ = fmin_ * 0.95
    
    N_ = (Vin_max / 2) / (Vout_)
    Nps = rational_opt(N_)
    N__ = round(Nps[0] / Nps[1], 2)
    
    M_set = N__ * Vout_max_ / (Vin_min_ / 2)
    
    f = np.linspace(fn0_(m_), fnb_(m_), 1000)
    M_pk = M_pk_fn(m_)(f)
    pon_pk = pon_pk_fn(m_)(f)
    
    M_set_id = np.argmin(abs(M_pk - M_set))
    f_set = f[M_set_id]
    pon_set = pon_pk[M_set_id]
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(f, M_pk, 'b', label=r'$m = 12$')
        ax1.set_xlabel(r'$f_N$')
        ax1.set_ylabel(r'$m = 12$')
        ax1.set_title(r'Maksimalna vrednost pojačanja $M_{MAX}(f_N)$')
        ax1.axhline(y=M_set, color='r', linestyle='--', label=r'$M_{SET}$')
        ax1.scatter(f_set, M_set, color='r', label=r'($f_{SET}$, $M_{SET}$)')
        
        ax2.plot(f, pon_pk, 'b', label=r'$P_{OUTn}$')
        ax2.set_xlabel(r'$f_N$')
        ax2.set_ylabel(r'$P_{OUTn}$')
        ax2.set_title(r'Maksimalna izlazna snaga $P_{OUTn}(f_N)$')
        ax2.scatter(f_set, pon_set, color='r', label=r'($f_{SET}$, $M_{SET}$)')
        
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
        
        fig.tight_layout()
        fig.savefig('mset_pon.pdf')
            
    fr_ = fmin__ / f_set
    wr_ = 2 * np.pi * fr_
        
    Zr_ = pon_set * N__**2 * Vout_max_ / Iout_max
    
    Lr_ = Zr_ / wr_
    Cr_ = 1 / (wr_**2 * Lr_)
    Lm_ = Lr_ * (m_ - 1)
    
    if printing:
        print("N = {:.2f}".format(N__))
        print("Lr = {:.2f} uH".format(Lr_ * 1e6))
        print("Cr = {:.2f} nF".format(Cr_ * 1e9))
        print("Lm = {:.2f} uH".format(Lm_ * 1e6))
        print("fr = {:.2f} kHz".format(fr_ * 1e-3))
    
    N__n = N__
    Lr_n = Lr_ * 1e6
    Cr_n = Cr_ * 1e9
    Lm_n = Lm_ * 1e6
    fr_n = fr_ / 1e3
    
    return (N__n, Lr_n, Cr_n, Lm_n, fr_n, Vout_max_)