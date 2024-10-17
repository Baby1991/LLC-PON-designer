import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from tqdm import tqdm

from design_library.basic_design import design_converter_with_m

#############################################################

def converter_limits(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_):
    
    m_s = np.linspace(4, 12, 100)

    Lr_s = []
    Lm_s = []
    Cr_s = []
    fr_s = []
    
    for m in m_s:
        N, Lr, Cr, Lm, fr, Vout_max = design_converter_with_m(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, m, fmin_)
        Lr_s.append(Lr)
        Lm_s.append(Lm)
        Cr_s.append(Cr)
        fr_s.append(fr)

    Cr1 = Cr_s[0]
    Cr2 = Cr_s[-1]

    Lr1 = Lr_s[0]
    Lr2 = Lr_s[-1]

    Lm1 = Lm_s[0]
    Lm2 = Lm_s[-1]

    fr1 = fr_s[0]
    fr2 = fr_s[-1]

    return ((Cr1, Cr2), (Lr1, Lr2), (Lm1, Lm2), (fr1, fr2))

#############################################################

def converter_limits_fmin(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_min, fmin_max, plot=False, npoints=100, show = True):

    Cr_min_s = []
    Cr_max_s = []
    Lr_min_s = []
    Lr_max_s = []
    Lm_min_s = []
    Lm_max_s = []
    fr_min_s = []
    fr_max_s = []

    f_s = np.linspace(fmin_min, fmin_max, npoints)

    for f in tqdm(f_s):
        ((Cr1, Cr2), (Lr1, Lr2), (Lm1, Lm2), (fr1, fr2)) = converter_limits(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, f)
        Cr_min_s.append(Cr1)
        Cr_max_s.append(Cr2)
        Lr_min_s.append(Lr1)
        Lr_max_s.append(Lr2)
        Lm_min_s.append(Lm1)
        Lm_max_s.append(Lm2)
        fr_min_s.append(fr1)
        fr_max_s.append(fr2)

    Cr_max = max(max(Cr_max_s), max(Cr_min_s))
    Cr_min = min(min(Cr_max_s), min(Cr_min_s))

    Cr_min_log = np.floor(np.log10(Cr_min))
    cap_s = np.array([1, 2.2, 3.3, 4.7, 6.8, 10, 22, 33, 47, 68, 100])

    #C1, C2 = np.meshgrid(cap_s, cap_s)
    #cap_s = np.unique(C1 + C2)
    
    cap_s = 10**Cr_min_log * cap_s
    cap_s = [cap for cap in cap_s if cap >= Cr_min and cap <= Cr_max]

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

        ax1.plot(f_s, Lr_min_s, 'r-', label='$L_R$($m = 4$)')
        ax1.plot(f_s, Lr_max_s, 'r--', label='$L_R$($m = 12$)')
        ax1.fill_between(f_s, Lr_min_s, Lr_max_s, color='r', alpha=0.2, label='$L_R$')

        ax1.plot(f_s, Lm_min_s, 'b-', label='$L_M$($m = 4$)')
        ax1.plot(f_s, Lm_max_s, 'b--', label='$L_M$($m = 12$)')
        ax1.fill_between(f_s, Lm_min_s, Lm_max_s, color='b', alpha=0.2, label='$L_M$')

        ax2.plot(f_s, Cr_min_s, 'r-', label='$C_R$($m = 4$)')
        ax2.plot(f_s, Cr_max_s, 'r--', label='$C_R$($m = 12$)')
        ax2.fill_between(f_s, Cr_min_s, Cr_max_s, color='r', alpha=0.2, label='$C_R$')

        ax3.plot(f_s, fr_min_s, 'r-', label='$f_R$($m = 4$)')
        ax3.plot(f_s, fr_max_s, 'r--', label='$f_R$($m = 12$)')
        ax3.fill_between(f_s, fr_min_s, fr_max_s, color='r', alpha=0.2, label='$f_R$')

        ax1.set_xlabel(r'$f_{min}$ [Hz]')
        ax2.set_xlabel(r'$f_{min}$ [Hz]')
        ax3.set_xlabel(r'$f_{min}$ [Hz]')
        ax1.set_ylabel(r'$L_R$ $L_M$ [µH]')
        ax2.set_ylabel(r'$C_R$ [nF]')
        ax3.set_ylabel(r'$f_R$ [kHz]')

        ax1.set_title(r'$L_R(f_{min}), L_M(f_{min})$')
        ax2.set_title(r'$C_R(f_{min})$')
        ax3.set_title(r'$f_R(f_{min})$')

        if cap_s:
            ax2.set_yticks(cap_s)

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.legend()
        ax2.legend()
        ax3.legend()

        ax1.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax2.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax3.set_xlim(fmin_min * 0.9, fmin_max * 1.1)

        fig.tight_layout()
        
        if plot and show:
            plt.show()

    return (f_s, (Cr_min_s, Cr_max_s), (Lr_min_s, Lr_max_s), (Lm_min_s, Lm_max_s), (fr_min_s, fr_max_s))

#############################################################