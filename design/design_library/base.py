import numpy as np
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
