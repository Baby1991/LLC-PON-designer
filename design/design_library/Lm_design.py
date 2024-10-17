import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from pqdm.processes import pqdm

from design_library.base import *
from design_library.basic_design import design_converter_with_m


#############################################################

def design_converter_Lm(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, Lm_, printing=False, savePrint=False):

    m_s = np.linspace(4, 12, 100)

    Lm_ = Lm_ * 1e6    

    if printing:
        print(40 * "*")
        print(f"Vin_min = {Vin_min} V")
        print(f"Vin_max = {Vin_max} V")
        print(f"Vout = {Vout_} V")
        print(f"Iout_max = {Iout_max} A")
        print(f"Vd_max = {Vd_max} V")
        print(f"fmin = {fmin_} Hz")
        print(40 * "-")

    if savePrint:
        output = []
        output.append(40 * "*")
        output.append(f"Vin_min = {Vin_min} V")
        output.append(f"Vin_max = {Vin_max} V")
        output.append(f"Vout = {Vout_} V")
        output.append(f"Iout_max = {Iout_max} A")
        output.append(f"Vd_max = {Vd_max} V")
        output.append(f"fmin = {fmin_} Hz")
        output.append(40 * "-")

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


    if printing:
        print("N = {:.2f}".format(N))
        print("Pout_max = {:.2f} W".format(Vout_ * Iout_max))
        print("Vout_max = {:.2f} V".format(Vout_max - Vd_max))
        print(40 * "*")
    if savePrint:
        output.append("N = {:.2f}".format(N))
        output.append("Pout_max = {:.2f} W".format(Vout_ * Iout_max))
        output.append("Vout_max = {:.2f} V".format(Vout_max - Vd_max))
        output.append(40 * "*")
    
    if min(Lm_s) > Lm_ or max(Lm_s) < Lm_:
        if printing:
            print("No optimal solution found")
            print("Lm_min = {:.2f} uH".format(min(Lm_s)))
            print("Lm_max = {:.2f} uH".format(max(Lm_s)))
            print(40 * "-")
        if savePrint:
            output.append("No optimal solution found")
            output.append("Lm_min = {:.2f} uH".format(min(Lm_s)))
            output.append("Lm_max = {:.2f} uH".format(max(Lm_s)))
            output.append(40 * "-")
        return
    
    closest = np.argmin(abs(np.array(Lm_s) - Lm_))

    m_best = m_s[closest]

    Lr_a = Lr_s[closest]
    Cr_a = Cr_s[closest]
    Lm_a = Lm_s[closest]
    fr_a = fr_s[closest]

    
        
    if printing:
        print("Lr = {:.0f} uH".format(Lr_a))
        print("Cr = {:.1f} nF".format(Cr_a))
        print("Lm = {:.0f} uH".format(Lm_a))
        print("fr = {:.2f} kHz".format(fr_a))
        print("m = {:.2f}".format(m_best))
        print(40 * "-")
    if savePrint:
        output.append("Lr = {:.0f} uH".format(Lr_a))
        output.append("Cr = {:.1f} nF".format(Cr_a))
        output.append("Lm = {:.0f} uH".format(Lm_a))
        output.append("fr = {:.2f} kHz".format(fr_a))
        output.append("m = {:.2f}".format(m_best))
        output.append(40 * "-")

    if savePrint:
        filename = f'{Vin_min}_{Vin_max}_{Vout_}_{Iout_max}_{Vd_max}_fmin={fmin_}_Lm={Lm_}'.replace('.', ',')
            
    if savePrint:
        with open(f'./Designs/{filename}.txt', 'w') as f:
            for line in output:
                f.write(line)
                f.write('\n')
            
    return ((Cr_a, Lr_a, Lm_a, fr_a), (Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, Lm_))


#############################################################

def design_converters_Lm(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_min, fmin_max, Lm_, printing=False, plot=False, savePrint=False, savePlot=False, npoints=100):

    if printing:
        raise Exception("design_converters_Lm: No printing available for multiple designs")

    Cr_s = {}
    Lr_s = {}
    Lm_s = {}
    fr_s = {}
    f_s_ = np.linspace(fmin_min, fmin_max, npoints)

    args = []
    for f in f_s_:
        args.append([Vin_min, Vin_max, Vout_, Iout_max, Vd_max, f, Lm_])

    Lm_ = Lm_ * 1e6

    all_solutions = pqdm(args, design_converter_Lm, n_jobs=8, argument_type='args')
        
    for i in range(len(all_solutions)):
        if all_solutions[i] is None:
            continue
        ((Cr_a, Lr_a, Lm_a, fr_a), func_input) = all_solutions[i]
        f = func_input[5]
        Cr_s[f] = Cr_a
        Lr_s[f] = Lr_a
        Lm_s[f] = Lm_a
        fr_s[f] = fr_a

    if savePlot or savePrint:
        filename = f'{Vin_min}_{Vin_max}_{Vout_}_{Iout_max}_{Vd_max}_{fmin_min}_{fmin_max}_{Lm_}'.replace('.', ',')

    if plot or savePlot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        
        ax1.scatter(Lr_s.keys(), Lr_s.values(), marker='.')
        ax1.scatter(Lm_s.keys(), Lm_s.values(), marker='.')
        ax2.scatter(Cr_s.keys(), Cr_s.values(), marker='.')
        ax3.scatter(fr_s.keys(), fr_s.values(), marker='.')

        ax1.set_xlabel(r'$f_{min}$ [Hz]')
        ax2.set_xlabel(r'$f_{min}$ [Hz]')
        ax3.set_xlabel(r'$f_{min}$ [Hz]')
        ax1.set_ylabel(r'$L_R$,$L_M$  [µH]')
        
        ax2.set_ylabel(r'$C_R$ [nF]')
        ax3.set_ylabel(r'$f_R$ [kHz]')

        ax1.set_title(r'$L_R(f_{min})$, $L_M(f_{min})$')
        ax2.set_title(r'$C_R(f_{min})$')
        ax3.set_title(r'$f_R(f_{min})$')

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax2.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax3.set_xlim(fmin_min * 0.9, fmin_max * 1.1)

        fig.tight_layout()
        
        if plot:
            plt.show()

        if savePlot:
            fig.savefig(f'./Designs/{filename}.pdf')

    if savePrint:
        with open(f'./Designs/{filename}.txt', 'w') as f:
            f.write(f"Vin_min = {Vin_min} V\n")
            f.write(f"Vin_max = {Vin_max} V\n")
            f.write(f"Vout = {Vout_} V\n")
            f.write(f"Iout_max = {Iout_max} A\n")
            f.write(f"Vd_max = {Vd_max} V\n")
            f.write(f"fmin_min = {fmin_min} Hz\n")
            f.write(f"fmin_max = {fmin_max} Hz\n")
            f.write((40 * "-") + "\n")
            f.write("fmin[Hz] Lr[uH] Lm[uH] Cr[nF] fr[kHz]\n")
            for f in Lr_s: 
                f.write(f"{f/1e3:.2f} {Lr_s[f]:.2f} {Lm_s[f]:.2f} {Cr_s[f]:.2f} {fr_s[f]:.2f}\n")
                f.write((40 * "*") + "\n")

    return (Cr_s, Lr_s, Lm_s, fr_s)

#############################################################