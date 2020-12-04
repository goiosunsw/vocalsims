#
# croak_poleres.py
#
"""
This module defines the vector field for a spring-mass reed 
couled to a single-mode resonator
"""

from numpy import sqrt, sign, exp, abs, pi, vstack, real, imag, floor, ceil, percentile, mod, log, empty, arange, frompyfunc, zeros
from numpy.random import rand
from math import isinf
from scipy.integrate import ode
from pylab import figure,plot,show,title,legend,subplot, ylim, ylabel, xlabel
import sys


debug = False

def sigmoid(x):
    if x > 50.0:
        return 0.0
    elif x < -50.0: 
        return 1.0
    else:
        return 1.0/(1.0 + exp(x))
        
# Second-order systems:
# 
# 1/w0^2 d2y/dt^2 + 1/(w0 Q0) dy/dt + 1 = a x
# (reed) or
#
# 1/w0^2 d2y/dt^2 + 1/(w0 Q0) dy/dt + 1 = a dx/dt
# (acoustic)
# 
# may be solved with a set of 2 first order systems
#
# y = y1 + y2
#
# in which
# 
# dy1/dt = s  y1 + c  x
# dy2/dt = s* y2 + c* x
#        
        
def modal_to_poleres_reed(w, q, a=1):
    """
    modal_to_poleres(w, q, a=1)
    Convert reed modal parameters to pole, residue.
    Parameters: 
        * w = angular frequency omega
        * q = q-factor
        * a = coupling amplitude (1 by default)
    """ 
    
    if q<=0.5:
        sys.stderr.write('Warning, reed is subcritical! Use q>0.5')
    
    inv2q = 1. / (2.*q)
    inv2q2 = inv2q*inv2q
    s = w * (-inv2q + 1j * sqrt(1. + 0j- inv2q2))
    c = -1j * a * w / (2. *  sqrt(1. + 0j - inv2q2))
    return s,c
     
def modal_to_poleres_acoust(w, q, a=1):
    """
    modal_to_poleres(w, q, a=1)
    Convert acoustic modal parameters to pole, residue.
    Parameters: 
        * w = angular frequency omega
        * q = q-factor
        * a = coupling amplitude (1 by default)
    """ 
    
    if q<=0.5:
        sys.stderr.write('Warning, acoustic mode is subcritical! Use q>0.5')
    
    
    inv2q = 1. / (2.*q)
    inv2q2 = inv2q*inv2q
    s = w * (-inv2q + 1j * sqrt(1. + 0j - inv2q2))
    ssc = s - s.conj()
    c = a * w / q * s / ssc
    #c = a * w  / q / 2. * (1. - 1j/(2.*q * sqrt(1. + 0j - inv2q2)))
    return s,c

class nlstiff(object):
    """
    Parent non-linear stiffness class
    Corresponds to linear stiffness (returns 0)
    
    Usage:  nl = nlstiff; f = nl(x)
    """     
    def __init__(self):
        self.f = lambda x: 0.0
    
    def __call__(self,x):
        return self.f(x)
     
class nlstiff_chatzi(nlstiff):
    """
    Non-linear stiffness defined by Chatziioanou
    (V. Chatziioannou, M. van Walstijn, 
     "Estimation of clarinet reed parameters by inverse modelling", 
     Act. Acust. un. Acust. Jul 2012, 98(4),629-639)
    
    Usage:  nl = nlstiff_chatzi(x_st, x_ev); f = nl(x)
        -x_st = position at which non-linear stiffness starts having an effect
        -x_ev = position at which nl component is the same as linear
    """     
    def __init__(self,x_st=0.6,x_ev=1.2):
        self.xc=x_st
        self.xs=x_ev
    
    def f(self,x):
        xdiff = -x-self.xc
        if xdiff > 0.0:
            return log((6.4*xdiff/(self.xs-self.xc))+1)
        else:
            return 0.0

class nlstiff_parabolic(nlstiff):
    """
    Non-linear parabolic stiffness 
    (similar to Chatziionnou  at low displacements but faster to calculate)
    
    Usage:  nl = nlstiff_parabolic(x_st, x_ev); f = nl(x)
        -x_st = position at which non-linear stiffness starts having an effect
        -x_ev = position at which nl component is the same as linear
    """     
    
    def __init__(self,x_st=0.6,x_ev=1.2):
        self.xc=x_st
        self.xs=x_ev
    
    def f(self,x):
        xdiff = -x-self.xc
        if xdiff > 0.0:
            return ((x+self.xc)/(self.xs-self.xc))**2
        else:
            return 0.0

class acoustic_mode(object):
    """
    Contains a generic acoustic mode 
    """
    
    def __init__(self, f = 1500.0, q = 20.0, a = None):
        self.f=f
        self.q=q
        if a is None:
            self.a = q
        else:
            self.a = a
        
        
        
        
    

class acoustic_mode_poleres(acoustic_mode):
    """
    Contains the description of an acoustic mode in pole/residues 
    """
    def __init__(self, f = 1500.0, q = 20.0, a = None):
        acoustic_mode.__init__(self,f,q,a=a)
        self.set_modal_params()
        self.pc = 0
        
    def calc_dp(self, u, pc):
        """
        Calculates the complex pressure derviative,
        given the current values of flow (u) and complex pressure (pc).
        Value of complex pressure is stored 
        """
        self.pc=pc
        return self.c*u + self.s*self.pc
    
    def set_modal_params(self):
        """
        Calculates the working modal parameters from the physical parameters
        """
    
        self.w  = 2*pi*self.f
        self.w2 = self.w*self.w
        
        self.s, self.c = modal_to_poleres_acoust(self.w, self.q, a=self.a)
        
    def set_initial_state(self, p0 = 0.0, dp0 = 0.0, u0=0):
        """
        Set the initial values of the state vector, for initial real values:
        Arguments
            * p0  : initial resonator pressure (1)
            * u0  : initial acoustic flow (0)
            * dp0 : starting derivative of pressure (0)
        Returns the complex pressure
        """
        
        # not sure why there are 3 input variables
        
        # pcr, pci: complex components of pressure vector pc
        pcr = p0/2.
        pci = (p0 * real(self.s) + u0 * real(self.c) - dp0/2) / imag(self.s)
        
        self.p0 = pcr + 1j*pci
        return self.p0
    
    def set_equilibrium_state(self, u0=0.0):
        """
        Set the initial values to an equilibrium point:
        """
        
        
        self.p0 = -self.c * u0 / self.s
        return self.p0

class croak_poleres(object):
    """
    Contains a reed crow simulator, based on a pole /residue approach
    """

    def __init__(self, gamma=0.5, zeta = 0.5, 
                 fr = 1500, qr = 5.0, ar=None, 
                 f1 = 1000, q1 = 30.0, a1=None, 
                 fv = None, qv=None, av=None,
                 nlfunc = None):
        """
        Return a crow simulator object
        Parameters:
        Exciter: 
            * gamma  = Non-dimensional mouth pressure (0-1)
            * zeta   = "Embouchure parameter" (0-1)
        Instrument acoustic resonator:
            * f1     = Resonance frequency
            * q1     = Q-factor
            * a1     = p/u coupling (default = 2 q1 / pi)
        Vocal tract acoustic resonator:
            * fv     = Resonance frequency
            * qv     = Q-factor
            * av     = p/u coupling (default = 2 q1 / pi)
        Reed oscillator: 
            * fr     = Resonance frequency
            * qr     = Q-factor
            * ar     = p/u coupling (default = 1)
            * nltog  = Boolean specifying the use of a non-linear stiffness (contact force)
        """
        
        if hasattr(gamma,'__call__'):
            self.gamma = gamma
        else:
            self.gamma = lambda x: gamma
            
        self.zeta = zeta
        self.fr = fr
        self.fa = f1
        self.fv = fv
        self.qr = qr
        self.qa = q1
        self.qv = qv
        if ar is None:
            self.ar = 1.0
        else:
            self.ar = ar
        if a1 is None:
            self.aa = q1
        else:
            self.aa = a1
        
        if self.fv is not None:
            if av is None:
                self.av = qv
            else:
                self.av = av
        # set mode amplitudes 
        self.set_modal_params()
        
        # viscous loss coefficient in the reed channel
        # FIX: should be ~ 80 L nu / 2 D^2  
        self.alpha  = .001
        self.alpha2 = self.alpha*self.alpha

        
        #  non-linear stiffness term
        #nltog = 1.0
        self.nlfunc = nlfunc
        if self.nlfunc is not None:
            from scipy.interpolate import interp1d
            x = arange(-2,5,0.01)
            nl = frompyfunc(nlfunc,1,1)
            dp = (1+nl(x))*x
            self.invnl = interp1d(dp,x)
        
        self.eps=1e-8
        
        if isinf(fr):
            self.dt = 0.01/f1
        else:
            self.dt = 0.01/max(max(f1),fr)
        self.tmax = 20.*self.dt
        self.set_t0(0.0)
        
        self.set_initial_state()
        self.error = False
        
    def set_tmax(self,tmax):
        self.tmax = tmax
        
    def set_dt(self,dt):
        self.dt = dt

    def get_dt(self):
        return self.dt

    def get_time(self):
        return self.cur_t

    def set_t0(self,t0):
        self.t0 = t0
        self.cur_t = t0

        
    def set_modal_params(self):
        #self.ar = 1.
        #self.a1 = self.q1
        
        self.wr  = 2*pi*self.fr
        self.wr2 = self.wr*self.wr
        
        self.wa  = 2*pi*self.fa
        self.wa2 = self.wa*self.wa

        
        # calculate poles and residues
        if not isinf(self.fr):
            self.sr, self.cr = modal_to_poleres_reed(self.wr, self.qr, a=self.ar)
        else:
            # claculate some modal parameters for compatibility
            # they will not be used
            self.sr, self.cr = modal_to_poleres_reed(1.0, self.qr, a=self.ar)
            
        self.acmode = []
        for thisf,thisq,thisa in zip(self.fa,self.qa,self.aa):
            #self.s1, self.c1 = modal_to_poleres_acoust(self.w1, self.q1, a=self.a1)
            self.acmode.append(acoustic_mode_poleres(thisf,thisq,thisa))
        self.nmodes = len(self.acmode)
        
        self.vtmode = []

        if self.fv is not None:
            self.wv  = 2*pi*self.fv
            self.wv2 = self.wv*self.wv
        
            for thisf,thisq,thisa in zip(self.fv,self.qv,self.av):
                #self.s1, self.c1 = modal_to_poleres_acoust(self.w1, self.q1, a=self.a1)
                self.vtmode.append(acoustic_mode_poleres(thisf,thisq,thisa))

        self.nvtmodes = len(self.vtmode)

    def vectorfield(self, t, w):
        """
        Defines the differential equations for the coupled reed-resonator system.
    
        Arguments:
            w :  vector of the state variables:
                      w = [x,p]
                xc: reed position (complex such that x=real(xc))
                pc: resonator pressure (complex such that p=real(pc))
            t :  times
        """
        
        gamma = self.gamma(t)
        xc = w[0]
        pc = w[1:self.nmodes+1]

        #sys.stderr.write('xc=%f, pc=%f\n' % (xc,pc))
        #sys.stderr.write('t=%g, x=%g, p=%g\n' % (t,2*real(xc),2*real(pc)))
        #gamma, zeta, w1, q1, f1, wr, qr, fr, nltog = self.par
        #eps = self.eps
        
        p = sum(2*real(pc))
        if self.nvtmodes>0:
            pvc = w[self.nmodes+1:self.nmodes+1+self.nvtmodes]
            pv = sum(2*real(pvc))
        else:
            pv = 0.

        # add all partial modes to build the total pressure
        #for mode in self.acmodes:
        #    p += 2.*real(pc[n])
            
        x = 2.*real(xc)
        
        sqrgp = sqrt(abs(gamma+pv-p)+self.alpha2) 


        u = self.zeta * (x+1.0) * (sqrgp-self.alpha) * sign(gamma + pv - p)
        
        if x < -1.0:
            #x=-1.0
            u = 0.0
        
        # solve the eqations for each acoustic mode
        f = empty(self.nmodes+self.nvtmodes+1,dtype='complex')
        for n,mode in enumerate(self.acmode):
            f[n+1]=mode.calc_dp(u,pc[n])

        if self.nvtmodes>0:
            n0 = n+2
            for n,mode in enumerate(self.vtmode):
                f[n+n0]=mode.calc_dp(-u,pvc[n])

        # add the reed equation
        if self.nlfunc is None:
            # Case : no contact force
            # Create f = (xc', pc'):
            #f = [(p-self.gamma)*self.cr +self.sr*xc,
            #     self.c1*u + self.s1*pc]
            f[0] = (p-pv-gamma)*self.cr +self.sr*xc
        else:
            # Create f = (xc', pc'):
            #f = [(p-self.gamma)/(1. + self.nlfunc(x))*self.cr +self.sr*xc,
            #     self.c1*u + self.s1*pc]
            f[0] = (p-pv-gamma)/(1. + self.nlfunc(x))*self.cr +self.sr*xc
                
                     
                     
        if sum(abs(f))==0.0:
            # just in case, this should'nt occur
            print ('Ouch!!! 0-valued vector!')
    
        if debug:
            print (t,x)
            #for ipar in par
            #   print >> sys.stderr , ipar
    
        return f

    def vf_acoustic_res(self, t, w):
        """
        Defines the differential equations for the coupled reed-resonator system.
    
        Arguments:
            w :  vector of the state variables:
                      w = [x,p]
                xc: reed position (complex such that x=real(xc))
                pc: resonator pressure (complex such that p=real(pc))
            t :  times
        """
        
        u = self.ufunc(t)
        
        xc = w[0]
        pc = w[1:]
        #sys.stderr.write('xc=%f, pc=%f\n' % (xc,pc))
        #sys.stderr.write('t=%g, x=%g, p=%g\n' % (t,2*real(xc),2*real(pc)))
        #gamma, zeta, w1, q1, f1, wr, qr, fr, nltog = self.par
        #eps = self.eps
        
        p = sum(2*real(pc))
        # add all partial modes to build the total pressure
        #for mode in self.acmodes:
        #    p += 2.*real(pc[n])
            
        # solve the eqations for each acoustic mode
        f = empty(self.nmodes+self.nvtmodes+1,dtype='complex')
        for n,mode in enumerate(self.acmode):
            f[n+1]=mode.calc_dp(u,pc[n])

        n0=n+2
        for n, mode in enumerate(self.vtmode):
            pvc = w[n+n0]
            f[n+n0] = mode.calc_dp(-u,pvc)
        
        # add the reed equation
        f[0] = 0.0
                     
                     
        if sum(abs(f))==0.0:
            # just in case, this should'nt occur
            print ('Ouch!!! 0-valued vector!')
    
        if debug:
            print (t,p)
            #for ipar in par
            #   print >> sys.stderr , ipar
        
        self.u = u
        
        return f


    def vectorfield_nodyn(self, t, w):
        """
        Defines the differential equations for the coupled reed-resonator system,
        without reed dynamics (reed follows the force immediately)
    
        Arguments:
            w :  vector of the state variables:
                      w = [pc,xc]
                xc: reed position (complex such that x=real(xc))
                    (maintained for compatibility, not used in calc)
                pc: resonator pressure (complex such that p=real(pc))
            t :  times
        """
        
        xc = w[0]
        pc = w[1:self.nmodes+1]
        #pc = w[1:]
        
        p = sum(2*real(pc))
        if self.nvtmodes>0:
            pvc = w[self.nmodes+1:self.nmodes+1+self.nvtmodes]
            pv = sum(2*real(pvc))
        else:
            pv = 0.

        gamma = self.gamma(t)
            
        x = 2.*real(xc)

        if self.nlfunc is None:
            x = (p-pv-gamma)
        else:
            x = self.invnl(p-pv-gamma) + 0.0

        
        sqrgp = sqrt(abs(gamma-p)+self.alpha2) 
        u = self.zeta * (x+1.0) * (sqrgp-self.alpha) * sign(gamma - p)

        if x < -1.0:
            #x=-1.0
            u = 0.0

        
        # solve the eqations for each acoustic mode
        f = empty(self.nmodes+self.nvtmodes+1,dtype='complex')
        for n,mode in enumerate(self.acmode):
            f[n+1]=mode.calc_dp(u,pc[n])
        
        if self.nvtmodes>0:
            n0 = n+2
            for n,mode in enumerate(self.vtmode):
                f[n+n0]=mode.calc_dp(-u,pvc[n])

        f[0] = 0.0
                     
        if sum(abs(f))==0.0:
            # just in case, this should'nt occur
            print ('Ouch!!! 0-valued vector!')
    
        if debug:
            print (t,x)
            #for ipar in par
            #   print >> sys.stderr , ipar
    
        return f

        
    def vf_reed(self, t, w):
        """
        Defines the differential equations for a forced reed alone.
    
        Arguments:
            w :  vector of the state variables:
                      w = [x]
                xc: reed position (complex such that x=real(xc))
            t :  times
        """
        
        xc = w
        #gamma, zeta, w1, q1, f1, wr, qr, fr, nltog = self.par
        
        x = 2*real(xc)

        # change here for external force    
        force = 0.0
        
        if self.reed_f_type == 'square':
            force = floor(mod(t*100,2))*2-1.0
        elif self.reed_f_type == 'rect':
            if t< 0.01:
                force = 1

        #sys.stderr.write('xc=%f, pc=%f\n' % (xc,pc))
        sys.stderr.write('t=%g, f=%g, x=%g\n' % (t,force,2*real(xc)))
                

        
        if self.nltog:
            # Case: Using non-linear stiffness / contact force
            nlterm = self.wr2/self.nlc  * sigmoid((x+1.0)/0.1)
            #nlterm =  sigmoid((x+1.0)/0.1)
            
            
            # Create f = (xc', pc'):
            f = [force*self.cr +self.sr*xc]
            
        else:
            # Case : no contact force
            
            # Characteristic non-linear function
            if x < -1.0:
                pass 
                               
            # Create f = (xc', pc'):
            f = [force*self.cr +self.sr*xc]
                     
                     
        if sum(abs(f))==0.0:
            # just in case, this should'nt occur
            print ('Ouch!!! 0-valued vector!')
    
        return f

        
    def set_initial_state(self, x0 = 0., v0 = 0., p0 = 0.0, dp0 = 0., pv0=0.0, dpv0=0.0):
        """
        Set the initial values of the state vector:
        Arguments
            * x0  : initial reed displacement (0)
            * v0  : initial reed velocity (0)
            * p0  : initial resonator pressure (1)
            * dp0 : starting derivative of pressure (0)
            * pv0  : initial vocal tract pressure (1)
            * dpv0 : starting derivative of vt pressure (0)
        """
        
        gamma = self.gamma(0.0)
        
        # xcr, xci: complex components of position vector xc
        xcr = x0/2.
        sri = imag(self.sr)
        if sri==0:
            xci = 0.
        else:
            xci = (x0 * real(self.sr) - v0) / (2. * sri)
        
        self.x0 = xcr + 1j*xci
        
        # initial flow is needed to calculate imaginary component of pressure
        if x0>-1:
            sqrgp = sqrt(abs(gamma-p0)+self.alpha2) 
            u0 = self.zeta * (x0+1) * sqrgp
        else:
            u0 = 0    
        
        # pcr, pci: complex components of pressure vector pc

        self.w0 = [self.x0]
        for mode in self.acmode:
            self.w0.append(mode.set_initial_state(p0=p0/self.nmodes,dp0=dp0,u0=u0))
            
        for mode in self.vtmode:
            self.w0.append(mode.set_initial_state(p0=pv0/self.nvtmodes,dp0=dpv0,u0=u0))

        #self.w0 = [self.x0, self.p0]
        
    def get_equilibrium_state(self, maxit=100, itstop=1e-8):
        """
        Get the iequilibrium state of the system:
        * maxit: maimum number of iterations 
        """
        
        gamma = self.gamma(0.0)
        
        # xcr, xci: complex components of position vector xc
        
        nit=0
        p0=0.0; dp=0.0; x0=0.0;v0=0;dp0=0
        dv=1.0
        while dv > itstop and nit<maxit:
            nit+=1
            p0prev = p0
            x0prev = x0
            
            xcr = real(x0)/2
            sri = imag(self.sr)
            if sri==0:
                xci = 0.
            else:
                xci = (x0 * real(self.sr) - v0) / (2. * sri)
        
            xc = xcr + 1j*xci
            x0 = 2*real(xc)
            
            if x0>-1:
                sqrgp = sqrt(abs(gamma-p0)+self.alpha2) 
                u0 = self.zeta * (x0+1) * sqrgp
            else:
                u0 = 0.0    
            
            pc = []
            for mode in self.acmode:
                pc.append(mode.set_equilibrium_state(u0=u0))
            p0 = sum(2*real(pc))
            x0 = p0-gamma
            
            dv = abs(x0-x0prev)+abs(p0-p0prev)
            
        return {'x': x0, 'p': p0, 'u': u0, 'nit': nit}
        
    def set_equilibrium_state(self, pert=0.0, maxit = 100):
        """
        Set the initial values of the state vector to eqilibrium:
        """
        
        gamma = self.gamma(0.0)
        
        # xcr, xci: complex components of position vector xc
        xcr = x0/2.
        sri = imag(self.sr)
        if sri==0:
            xci = 0.
        else:
            xci = (x0 * real(self.sr) - v0) / (2. * sri)
        
        self.x0 = xcr + 1j*xci
        
        # initial flow is needed to calculate imaginary component of pressure
        if x0>-1:
            sqrgp = sqrt(abs(gamma-p0)+self.alpha2) 
            u0 = self.zeta * (x0+1) * sqrgp
        else:
            u0 = 0    
        
        # pcr, pci: complex components of pressure vector pc

        self.w0 = [self.x0]
        for mode in self.acmode:
            self.w0.append(mode.set_equilibrium_state(p0=p0/self.nmodes,dp0=dp0,u0=u0))
            
        #self.w0 = [self.x0, self.p0]

    def ac_impulse_response(self):
        """
        Return th reed response to a signal 
        Arguments
            * fsig Forcing signal (default: impulse);
        """
        
        w0=zeros(self.nmodes+self.nvtmodes+1)
        
        odesol = ode(f=self.vf_acoustic_res)
        odesol.set_integrator('zvode',nsteps=500)
        odesol.set_initial_value(w0,self.t0)
        
        tsol=[]
        wsola=[]
        uf = []
        p = []
        pv = []
        
        # Call the ODE solver.
        #wsol = odeint(croak.vectorfield, w0, t, args=(par,),
        #              atol=abserr, rtol=relerr)
        
        #odesol.set_f_params(0)
        #self.u=1.0
        



        def pointwise(x):
            if x < 0.0:
                return 0.0
            elif x > self.dt*0.5:
                return 0.0
            else:
                return 1.0
            

        self.ufunc=frompyfunc(pointwise,1,1)
        odesol.integrate(odesol.t)
        tsol.append(odesol.t)
        wsola.append(2*real(odesol.y))
        uf.append(self.u)
        p.append(sum(2*real(odesol.y[1:self.nmodes+1])))
        pv.append(sum(2*real(odesol.y[self.nmodes+1:1+self.nmodes+self.nvtmodes])))
        while odesol.t <= self.tmax:
            
            if odesol.successful():
                odesol.integrate(odesol.t + self.dt)
                tsol.append(odesol.t)
                wsola.append(2*real(odesol.y))
                uf.append(self.u)
                p.append(sum(2*real(odesol.y[1:self.nmodes+1])))
                pv.append(sum(2*real(odesol.y[self.nmodes+1:1+self.nmodes+self.nvtmodes])))
            else:
                sys.stderr.write('!!!!! Integration STOPPED prematurely!\n')
                sys.stderr.write('Return code: %d'%odesol.get_return_code())
                break
                
        t = tsol
        u = vstack(uf)
        y = (vstack(wsola))
        
        return u.squeeze(), vstack(p).squeeze(),vstack(pv).squeeze() 

    def ac_noise_response(self,tmax=0.5):
        """
        Return th reed response to a signal 
        Arguments
            * fsig Forcing signal (default: impulse);
        """
        
        w0=zeros(self.nmodes+self.nvtmodes+1)
        
        odesol = ode(f=self.vf_acoustic_res)
        odesol.set_integrator('zvode',nsteps=500)
        odesol.set_initial_value(w0,self.t0)
        
        tsol=[]
        wsola=[]
        uf = []
        p = []
        pv = []
        
        # Call the ODE solver.
        #wsol = odeint(croak.vectorfield, w0, t, args=(par,),
        #              atol=abserr, rtol=relerr)
        
        #odesol.set_f_params(0)
        #self.u=1.0
        while odesol.t <= tmax:
            self.u = rand()-0.5
            if odesol.successful():
                odesol.integrate(odesol.t + self.dt)
                tsol.append(odesol.t)
                wsola.append(2*real(odesol.y))
                uf.append(self.u)
                p.append(sum(2*real(odesol.y[1:self.nmodes+1])))
                pv.append(sum(2*real(odesol.y[self.nmodes+1:1+self.nmodes+self.nvtmodes])))
            else:
                sys.stderr.write('!!!!! Integration STOPPED prematurely!\n')
                sys.stderr.write('Return code: %d'%odesol.get_return_code())
                break
            #self.u=0.0
        
        t = tsol
        u = vstack(uf)
        y = (vstack(wsola))
        
        return u.squeeze(), vstack(p).squeeze(), vstack(pv).squeeze()


    
    def force_reed(self, ftype = 'square'):
        """
        Return th reed response to a signal 
        Arguments
            * fsig Forcing signal (default: impulse);
        """
        
        self.reed_f_type = ftype
        
        w0=[0.0]
        
        odesol = ode(f=self.vf_reed)
        odesol.set_integrator('zvode',nsteps=500)
        odesol.set_initial_value(w0,self.t0)
        
        tsol=[]
        wsola=[]
        
        # Call the ODE solver.
        #wsol = odeint(croak.vectorfield, w0, t, args=(par,),
        #              atol=abserr, rtol=relerr)
        
        #odesol.set_f_params(0)
        
        while odesol.t <= self.tmax:
            if odesol.successful():
                odesol.integrate(odesol.t + self.dt)
                tsol.append(odesol.t)
                wsola.append(2*real(odesol.y))
            else:
                sys.stderr.write('!!!!! Integration STOPPED prematurely!\n')
                sys.stdout.write('!!!!! Integration STOPPED prematurely!\n')
                break
        
        t = tsol
        y = vstack(wsola)

        figure()
        plot(t,y)
        show()
    
    def setup_integrator(self):
        """
        Sets up the ode integrator with initial conditions
        """
        #self.set_initial_state()
        
        if isinf(self.fr):
            self.odesol = ode(f=self.vectorfield_nodyn)
        else:
            self.odesol = ode(f=self.vectorfield)
        self.odesol.set_integrator('zvode',nsteps=500)
        self.odesol.set_initial_value(self.w0,self.t0)
        
        self.tsol=[]
        self.wsola=[]
 
    def calculate_at_time(self, t):
        """
        Calculates a new value of the soltion at time t
        """
        if t>self.cur_t:
            if self.odesol.successful():
                self.odesol.integrate(t)
                self.tsol.append(self.odesol.t)
                p = sum(2*real(self.odesol.y[1:1+self.nmodes]))
                if self.nvtmodes>0:
                    pv = sum(2*real(self.odesol.y[1+self.nmodes:1+self.nmodes+self.nvtmodes]))
                else:
                    pv = 0.
                gamma = self.gamma(self.odesol.t)
                if isinf(self.fr):
                    if self.nlfunc is None:
                        x = p-pv-gamma
                    else:
                        x = self.invnl(p-pv-gamma)+0.0
                        #x = (p-pv-gamma)/(1. + self.nlfunc(x))
                else:
                    x = (2*real(self.odesol.y[0]))
                
                thissol = [x,p,pv]
                self.wsola.append(thissol)
            else:
                sys.stderr.write('!!!!! Integration STOPPED prematurely!\n')
                sys.stdout.write('!!!!! Integration STOPPED prematurely!\n')
                self.error=True
                
            self.cur_t = self.odesol.t
        else:
            sys.stderr.write('Illegal time value!!!\n')
            sys.stderr.write('Last time = %f, requested time = %f\n'%(self.cur_t, t))

    def integrate(self):
        """
        Solve the differential equation 
        Arguments
            * w0 = [x0, v0, p0, dp0] : initial vector
        """
        
        self.setup_integrator()
        # Call the ODE solver.
        #wsol = odeint(croak.vectorfield, w0, t, args=(par,),
        #              atol=abserr, rtol=relerr)
        
        #odesol.set_f_params(0)
        
        while self.odesol.t <= self.tmax and not self.error:
            self.calculate_at_time(self.cur_t+self.dt)
            
        self.finish_simulation()
        
    def finish_simulation(self):
        self.t = self.tsol
        self.y = vstack(self.wsola)
    
    def plot_sols(self,plotvar=0):
        
        t = self.t
        
        figure();

        ax1 = subplot(2,1,1)
        plot(self.t,self.y[:,0], axes=ax1)
        ylim(floor(percentile(self.y[:,0],5)),ceil(percentile(self.y[:,0],95)))
        title('$f_1$=%.0f; $Q_1$=%.1f; $f_r$=%.1f; $Q_r$=%.1f; $\gamma$=%.2f; $\zeta$=%.2f'%(self.f1,self.q1,self.fr,self.qr,self.gamma,self.zeta))
        ylabel('Reed displacement')
        
        ax2 = subplot(2,1,2,sharex=ax1)
        plot(self.t,self.y[:,1], axes=ax2)
        ylim(floor(percentile(self.y[:,1],5)),ceil(percentile(self.y[:,1],95)))
        ylabel('Acoust. pressure')
        xlabel('Time')
        
        #legend(names)
        show()
    
