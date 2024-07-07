import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from numpy import sin, cos, sqrt, pi

#############################################################

def M_b_PON_PN_fn(m_):
    def M_b_PON_PN(fn):
        return (m_-1) / sqrt(m_**2 - 2 * m_ * sin(pi / (2 * fn)) * (sin(pi / (2 * fn)) - pi / (2*fn) * cos(pi / (2 * fn))) + (sin(pi / (2 * fn)) - pi / (2*fn) * cos(pi / (2 * fn)))**2)
    return M_b_PON_PN    

#############################################################

def fnb_(m_):
    fns = np.linspace(0.7, 1, 10000)
    M_b_PON_PN_ = M_b_PON_PN_fn(m_)
    M_ = M_b_PON_PN_(fns)
    M_max_id = np.argmax(M_)
    fnb = fns[M_max_id]
    return fnb
    
#############################################################

def IrOnb_(m_):
    fnb = fnb_(m_)
    return np.sqrt((np.pi / 2 / fnb)**2 + m_) / (m_ - 1)

#############################################################

def IrOn0_(m_):
    return np.sqrt(m_) / (m_ - 1)

#############################################################

def fn0_(m_):
    return 1 / np.sqrt(m_)

#############################################################

def IrOn_fn(m_):
    IrOn0 = IrOn0_(m_)
    IrOnb = IrOnb_(m_)
    fnb = fnb_(m_)
    fn0 = fn0_(m_)
    return lambda fn: (IrOn0 - IrOnb) / (fn0 - fnb) * (fn - fnb) + IrOnb

#############################################################

def beta_fn(m_):
    IrOn_fn_ = IrOn_fn(m_)
    return lambda fn : np.arccos(-np.sqrt(m_) / (m_ - 1) / IrOn_fn_(fn))

#############################################################

def thetaO0_fn(m_):
    beta = beta_fn(m_)
    alpha = lambda fn : (-1 - np.cos(beta(fn)) * ((np.pi / (fn * np.sqrt(m_))) - beta(fn)) - np.sin(beta(fn))) / np.cos(beta(fn))
    return lambda fn : (np.cos(alpha(fn)) * alpha(fn) - np.sin(alpha(fn)) - np.cos(beta(fn)) * (np.pi / (fn * np.sqrt(m_)) - beta(fn)) - np.sin(beta(fn))) / (np.cos(alpha(fn)) + np.cos(beta(fn)))
    
#############################################################

def IrPn_fn(m_):
    IrOn_fn_ = IrOn_fn(m_)
    thetaO0_fn_ = thetaO0_fn(m_)
    return lambda fn : np.sqrt((IrOn_fn_(fn) * np.sin(thetaO0_fn_(fn)))**2 + (np.sqrt(m_) * IrOn_fn_(fn) * np.cos(thetaO0_fn_(fn)) - 1)**2)

def IrNn_fn(m_):
    IrOn_fn_ = IrOn_fn(m_)
    return lambda fn : np.sqrt(IrOn_fn_(fn) ** 2 - 1 / (m_ - 1))

#############################################################

def M_pk_fn(m_):
    IrPn_fn_ = IrPn_fn(m_)
    IrNn_fn_ = IrNn_fn(m_)
    return lambda fn : 2 / (IrPn_fn_(fn) - IrNn_fn_(fn))

#############################################################

def pon_pk_fn(m_):
    IrOn_fn_ = IrOn_fn(m_)
    thetaO0_fn_ = thetaO0_fn(m_)
    def pon_pk(fn):
        IrOn = IrOn_fn_(fn)
        thetaO0 = thetaO0_fn_(fn)
        return fn / pi * ( (m_ - 1) / 2 * (IrOn * cos(thetaO0))**2 - sqrt(m_) * IrOn * cos(thetaO0) 
                           + sqrt( (IrOn * sin(thetaO0))**2 + ( sqrt(m_) * IrOn * cos(thetaO0) - 1 )**2 )
                           - sqrt( IrOn**2 - 1 / (m_ - 1) ) + m_ / 2 / (m_ - 1)
                          ) 
    return pon_pk

#############################################################

def rational_opt(ratio, error = 0.1, max_q = 10):
    # p, q -> p/q
    
    if ratio > 0:
        if ratio > 1:
            
            for q in range(1, max_q+1):
                p = round(q * ratio)
                
                if abs(p/q - ratio) < error:
                    return [p, q]
                
            else:
                raise RecursionError('Max Iterations Reached')
                
        elif ratio < 1:
            return rational_opt(1/ratio, error=error, max_q=max_q)[::-1]
        else:
            return [1, 1]
        
    else:
        raise NotImplementedError('Zero and Negative ratios are not supported')
    
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
        ax1.plot(f, M_pk, 'b', label=r'$M_{MAX}$')
        ax1.set_xlabel(r'$f_N$')
        ax1.set_ylabel(r'$M_{MAX}$')
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

#############################################################

def design_converter(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_, printing=False, plot=False):
    
    m_s = np.linspace(4, 12, 100)
    m_n = np.arange(4, 12+1)
    
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
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
      
    if plot:  
        ax1.plot(m_s, Lr_s, 'r-', label='$L_R$')
        ax1.plot(m_s, Lm_s, 'b--',label='$L_M$')
        ax2.plot(m_s, Cr_s, 'r', label='$C_R$')
        ax3.plot(m_s, fr_s, 'r', label='$f_R$')

    Cr_min = min(Cr_s)
    Cr_max = max(Cr_s)
    Cr_min_log = np.floor(np.log10(Cr_min))
    cap_s = np.array([1, 2.2, 3.3, 4.7, 6.8, 10])
    cap_s = 10**Cr_min_log * cap_s
    cap_s = [cap for cap in cap_s if cap >= Cr_min and cap <= Cr_max]

    if printing:
        print("N = {:.2f}".format(N))
        print("Pout_max = {:.2f} W".format(Vout_ * Iout_max))
        print("Vout_max = {:.2f} V".format(Vout_max - Vd_max))
        print(40 * "-")
        
    optimals = []

    for cap in cap_s:
        closest = np.argmin(abs(np.array(Cr_s) - cap))
        optimals.append((m_s[closest], Lr_s[closest], Cr_s[closest], Lm_s[closest], fr_s[closest]))

        if plot:
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

    if cap_s == [] and printing:
        print("No optimal solution found")
        print("Cr_min = {:.2f} nF".format(Cr_min))
        print("Cr_max = {:.2f} nF".format(Cr_max))
        print(40 * "-")

    if plot:
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
        
        plt.show()
            
    return optimals

#############################################################

if __name__ == "__main__":
    design_converter(385, 420, 12, 12.5, 0.1, 90e3, printing=True, plot=True) # Good solution
    design_converter(385, 420, 12, 13,   0.1, 90e3, printing=True, plot=True) # Bad solution
