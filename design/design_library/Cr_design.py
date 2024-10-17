import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from pqdm.processes import pqdm

from design_library.base import *
from design_library.rational_opt import rational_opt
from design_library.basic_design import design_converter_with_m

#############################################################

def make_deltaCr_for_m(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, Cr__):
    Vout_max_d = (Vout_ + Vd_max)
    Vout_max_ = Vout_max_d * 1.10
    Vin_min_ = Vin_min * 0.95
    fmin__ = fmin_ * 0.95
    
    N_ = (Vin_max / 2) / (Vout_)
    Nps = rational_opt(N_)
    N__ = round(Nps[0] / Nps[1], 2)
    
    M_set = N__ * Vout_max_ / (Vin_min_ / 2)
    
    def deltaCr_for_m(m_):
        f = np.linspace(fn0_(m_), fnb_(m_), 1000)
        M_pk = M_pk_fn(m_)(f)
        pon_pk = pon_pk_fn(m_)(f)
        
        M_set_id = np.argmin(abs(M_pk - M_set))
        f_set = f[M_set_id]
        pon_set = pon_pk[M_set_id]
        
        fr_ = fmin__ / f_set
        wr_ = 2 * np.pi * fr_
            
        Zr_ = pon_set * N__**2 * Vout_max_ / Iout_max
        
        Lr_ = Zr_ / wr_
        Cr_ = 1 / (wr_**2 * Lr_)
    
        return (Cr_ * 1e9 - Cr__) / 10

    return deltaCr_for_m

#############################################################

def design_converter_Cr(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, Cr_pairs=False, askCr=False, printing=False, plot=False, savePrint=False, savePlot=False):
    
    m_s = np.linspace(4, 12, 100)
    m_n = np.arange(4, 12+1)
    
    if plot or savePlot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))    

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
      
    if plot or savePlot:  
        ax1.plot(m_s, Lr_s, 'r-', label='$L_R$')
        ax1.plot(m_s, Lm_s, 'b--',label='$L_M$')
        ax2.plot(m_s, Cr_s, 'r', label='$C_R$')
        ax3.plot(m_s, fr_s, 'r', label='$f_R$')

    Cr_min = 0.9 * min(Cr_s)
    Cr_max = 1.1 * max(Cr_s)

    if printing:
        print("N = {:.2f}".format(N))
        print("Pout_max = {:.2f} W".format(Vout_ * Iout_max))
        print("Vout_max = {:.2f} V".format(Vout_max - Vd_max))
        print("Cr_min = {:.1f} nF".format(Cr_min))
        print("Cr_max = {:.1f} nF".format(Cr_max))
        print(40 * "*")
    if savePrint:
        output.append("N = {:.2f}".format(N))
        output.append("Pout_max = {:.2f} W".format(Vout_ * Iout_max))
        output.append("Vout_max = {:.2f} V".format(Vout_max - Vd_max))
        output.append("Cr_min = {:.1f} nF".format(Cr_min))
        output.append("Cr_max = {:.1f} nF".format(Cr_max))
        output.append(40 * "*")
    
    Cr_min_log = np.floor(np.log10(Cr_min))
    cap_s = np.array([1, 2.2, 3.3, 4.7, 6.8, 10])

    if Cr_pairs:
        C1, C2 = np.meshgrid(cap_s, cap_s)
        cap_s_adds = np.unique(C1 + C2)
        cap_s = np.concatenate((cap_s, cap_s_adds))
        cap_s = np.sort(np.unique(cap_s))
    
    cap_s = 10**Cr_min_log * cap_s
    cap_s = [cap for cap in cap_s if cap >= Cr_min and cap <= Cr_max]
        
    optimals = []

    for cap in cap_s[::-1]:
        closest = np.argmin(abs(np.array(Cr_s) - cap))

        m__ = np.linspace(max(m_s[closest] - 0.1, 4), min(m_s[closest] + 0.1, 12), 200)

        deltaCr = abs(np.array([make_deltaCr_for_m(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, cap)(m) for m in m__]))

        closest1= np.argmin(deltaCr)
        m_best = m__[closest1]

        _, Lr, Cr, Lm, fr, _ = design_converter_with_m(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, m_best, fmin_)

        optimals.append((m_best, Lr, Cr, Lm, fr))

    if optimals:
        m_best, Lr, Cr, Lm, fr = optimals[0]

        if plot or savePlot:
            ax2.axvline(x=m_best, color='r', linestyle='--', label=r"optimal $m$")

            ax1.scatter(m_best, Lr, marker='s', c='r', label=f"$L_R$ solution")
            ax1.scatter(m_best, Lm, marker='o', c='b', label=f"$L_M$ solution")
            ax2.scatter(m_best, Cr, marker='s', c='r', label=f"$C_R$ solution")
            ax3.scatter(m_best, fr, marker='s', c='r', label=f"$f_R$ solution")
        
        if printing:
            print("Lr = {:.0f} uH".format(Lr))
            print("Cr = {:.1f} nF".format(Cr))
            print("Lm = {:.0f} uH".format(Lm))
            print("fr = {:.2f} kHz".format(fr))
            print("m = {:.2f}".format(m_best))
            print(40 * "-")
        if savePrint:
            output.append("Lr = {:.0f} uH".format(Lr))
            output.append("Cr = {:.1f} nF".format(Cr))
            output.append("Lm = {:.0f} uH".format(Lm))
            output.append("fr = {:.2f} kHz".format(fr))
            output.append("m = {:.2f}".format(m_best))
            output.append(40 * "-")

    
    for m_best, Lr, Cr, Lm, fr in optimals[1:]:
        if plot or savePlot:
            ax2.axvline(x=m_best, color='r', linestyle='--')

            ax1.scatter(m_best, Lr, marker='s', c='r')
            ax1.scatter(m_best, Lm, marker='o', c='b')
            ax2.scatter(m_best, Cr, marker='s', c='r')
            ax3.scatter(m_best, fr, marker='s', c='r')
        
        if printing:
            print("Lr = {:.0f} uH".format(Lr))
            print("Cr = {:.1f} nF".format(Cr))
            print("Lm = {:.0f} uH".format(Lm))
            print("fr = {:.2f} kHz".format(fr))
            print("m = {:.2f}".format(m_best))
            print(40 * "-")
        if savePrint:
            output.append("Lr = {:.0f} uH".format(Lr))
            output.append("Cr = {:.1f} nF".format(Cr))
            output.append("Lm = {:.0f} uH".format(Lm))
            output.append("fr = {:.2f} kHz".format(fr))
            output.append("m = {:.2f}".format(m_best))
            output.append(40 * "-")


    if cap_s == []:
        if printing:
            print("No optimal solution found")
            print("Cr_min = {:.2f} nF".format(Cr_min))
            print("Cr_max = {:.2f} nF".format(Cr_max))
            print(40 * "-")

            if askCr:
                cap = float(input("Enter Cr value: "))

                closest = np.argmin(abs(np.array(Cr_s) - cap))
                optimals.append((m_s[closest], Lr_s[closest], Cr_s[closest], Lm_s[closest], fr_s[closest]))

                if plot or savePlot:
                    ax2.axvline(x=m_s[closest], color='r', linestyle='--', label=r"optimal $m$")

                    ax1.scatter(m_s[closest], Lr_s[closest], marker='s', c='r', label=f"$L_R$ solution")
                    ax1.scatter(m_s[closest], Lm_s[closest], marker='o', c='b', label=f"$L_M$ solution")
                    ax2.scatter(m_s[closest], Cr_s[closest], marker='s', c='r', label=f"$C_R$ solution")
                    ax3.scatter(m_s[closest], fr_s[closest], marker='s', c='r', label=f"$f_R$ solution")
                
                if printing:
                    print("Lr = {:.0f} uH".format(Lr_s[closest]))
                    print("Cr = {:.1f} nF".format(Cr_s[closest]))
                    print("Lm = {:.0f} uH".format(Lm_s[closest]))
                    print("fr = {:.2f} kHz".format(fr_s[closest]))
                    print("m = {:.2f}".format(m_s[closest]))
                    print(40 * "-")
                if savePrint:
                    output.append("Lr = {:.0f} uH".format(Lr_s[closest]))
                    output.append("Cr = {:.1f} nF".format(Cr_s[closest]))
                    output.append("Lm = {:.0f} uH".format(Lm_s[closest]))
                    output.append("fr = {:.2f} kHz".format(fr_s[closest]))
                    output.append("m = {:.2f}".format(m_s[closest]))
                    output.append(40 * "-")


                
        if savePrint:
            output.append("No optimal solution found")
            output.append("Cr_min = {:.2f} nF".format(Cr_min))
            output.append("Cr_max = {:.2f} nF".format(Cr_max))
            output.append(40 * "-")

    if savePrint or savePlot:
        filename = f'{Vin_min}_{Vin_max}_{Vout_}_{Iout_max}_{Vd_max}_{fmin_}'.replace('.', ',')

    if plot or savePlot:
        ax1.set_xlabel(r'$m$')
        ax2.set_xlabel(r'$m$')
        ax3.set_xlabel(r'$m$')
        
        ax1.set_title(r'$L_R$(m) $L_M$(m) [µH]')
        ax1.set_xticks(m_n)
        ax2.set_title(r'$C_R$(m) [nF]')
        ax2.set_xticks(m_n)
        if cap_s:
            ax2.set_yticks(cap_s)
        ax3.set_title(r'$f_R$(m) [kHz]')
        ax3.set_xticks(m_n)

        ax1.legend()
        ax2.legend()
        ax3.legend()

        ax1.grid()
        ax2.grid()
        ax3.grid()
        
        fig.tight_layout()
        
        if plot:
            plt.show()
        
        if savePlot:
            fig.savefig(f'./Designs/{filename}.pdf')
            
    if savePrint:
        with open(f'./Designs/{filename}.txt', 'w') as f:
            for line in output:
                f.write(line)
                f.write('\n')
            
    return (optimals, (Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_))

#############################################################

def design_converters_Cr(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_min, fmin_max, Cr_pairs=False, plot=False, savePlot=False, save=False, npoints=100):

    Cr_s = {}
    Lr_s = {}
    Lm_s = {}
    fr_s = {}
    f_s_ = np.linspace(fmin_min, fmin_max, npoints)
    f_s = {}

    args = []
    for f in f_s_:
        args.append([Vin_min, Vin_max, Vout_, Iout_max, Vd_max, f, Cr_pairs])

    all_solutions = pqdm(args, design_converter_Cr, n_jobs=8, argument_type='args')

    for (solution_set, func_input) in all_solutions:
        for i in range(len(solution_set)):
            sol = solution_set[i]
            # m_s, Lr_s, Cr_s, Lm_s, fr_s
            if i not in Cr_s:
                Cr_s[i] = []
                Lr_s[i] = []
                Lm_s[i] = []
                fr_s[i] = []
                f_s[i] = []
            f_s[i].append(func_input[5])
            Cr_s[i].append(sol[2])
            Lr_s[i].append(sol[1])
            Lm_s[i].append(sol[3])
            fr_s[i].append(sol[4])

    if savePlot or save:
        filename = f'{Vin_min}_{Vin_max}_{Vout_}_{Iout_max}_{Vd_max}_{fmin_min}_{fmin_max}_pairs={Cr_pairs}'.replace('.', ',')

    if plot or savePlot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        for i in Cr_s:
            ax1.scatter(f_s[i], Lr_s[i], marker='.')
            ax2.scatter(f_s[i], Lm_s[i], marker='.')
            ax3.scatter(f_s[i], Cr_s[i], marker='.')
            ax4.scatter(f_s[i], fr_s[i], marker='.')

        ax1.set_xlabel(r'$f_{min}$ [Hz]')
        ax2.set_xlabel(r'$f_{min}$ [Hz]')
        ax3.set_xlabel(r'$f_{min}$ [Hz]')
        ax4.set_xlabel(r'$f_{min}$ [Hz]')
        ax1.set_ylabel(r'$L_R$ [µH]')
        ax2.set_ylabel(r'$L_M$ [µH]')
        ax3.set_ylabel(r'$C_R$ [nF]')
        ax4.set_ylabel(r'$f_R$ [kHz]')

        ax1.set_title(r'$L_R(f_{min})$')
        ax2.set_title(r'$L_M(f_{min})$')
        ax3.set_title(r'$C_R(f_{min})$')
        ax4.set_title(r'$f_R(f_{min})$')

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        #ax1.legend()
        #ax2.legend()
        #ax3.legend()
        #ax4.legend()

        ax1.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax2.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax3.set_xlim(fmin_min * 0.9, fmin_max * 1.1)
        ax4.set_xlim(fmin_min * 0.9, fmin_max * 1.1)

        fig.tight_layout()
        
        if plot:
            plt.show()

        if savePlot:
            fig.savefig(f'./Designs/{filename}.pdf')

    if save:
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
            for i, _ in f_s.items():
                for j in range(len(f_s[i])):
                    f.write(f"{f_s[i][j]/1e3:.2f} {Lr_s[i][j]:.2f} {Lm_s[i][j]:.2f} {Cr_s[i][j]:.2f} {fr_s[i][j]:.2f}\n")
                f.write((40 * "*") + "\n")

#############################################################