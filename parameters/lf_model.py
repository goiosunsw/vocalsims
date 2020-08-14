import numpy as np
import sympy as sp

E_0, E_1, alpha, w_g, t_e, t_c, epsilon, t = \
    sp.symbols('E_0 E_1 alpha w_g t_e t_c epsilon t', real=True)


class LFModel(object):
    def __init__(self, tp=0.3, ta=0.03, te=0.5, tc=1.0):

        forward_f = E_0 * sp.exp(alpha * t) * sp.sin(w_g*t)
        return_f = -E_1 * (-sp.exp(-epsilon*(t_c-t_e)) + sp.exp(-epsilon*(t-t_e)))

        # definition of t_p
        t_p = sp.symbols('t_p')
        self.tp_def = sp.pi/w_g

        # set initial parameters
        self.tp = tp
        self.ta = ta
        self.tc = tc
        self.te = te

        self._forward_f = forward_f
        self._return_f = return_f

        self.set_timing_parameters()


    def set_timing_parameters(self,tp=None,ta=None,te=None,tc=None,alpha_guess=2.):
        if tp is None:
            tp = self.tp
        if ta is None:
            ta = self.ta
        if te is None:
            te = self.te
        if tc is None:
            tc = self.tc
        # directly replaceable parameters
        par_sub = [(t_e,te),(t_c,tc),(E_0,1.)]

        wg_val = sp.nsolve(tp-self.tp_def,w_g,sp.N(sp.pi))
        print("w_g = ",wg_val)
        par_sub.append((w_g,wg_val))

        # solve for epsilon
        # (this is not exact, but almost if t_a<<t_c-t_e)
        epsilon_val = sp.nsolve(
            (1-sp.exp(-epsilon*(t_c-t_e))-epsilon*ta).subs(par_sub), epsilon, 1/ta)
        print("epsilon = ", epsilon_val)
        par_sub.append((epsilon, epsilon_val))


        return_int = sp.integrate(self._return_f, (t, t_e, t_c), conds='none')
        forward_int = sp.integrate(self._forward_f, (t, 0, t_e), conds='none')


        f = [(self._return_f-self._forward_f).subs(par_sub).subs(t, te),
            forward_int.subs(par_sub)+return_int.subs(par_sub)]
        E_1_guess = sp.sin(wg_val*te)*sp.exp(alpha_guess*te)
        sol = sp.nsolve(f, [alpha, E_1], [alpha_guess, E_1_guess], dict=True)[0]
        for k,v in sol.items():
            print(k," = ",v)

        sol_sub = par_sub
        sol_sub.extend([(k,v) for k,v in sol.items()])

        g = sp.re(sp.Piecewise((self._forward_f.subs(sol_sub),t<te),
                               (self._return_f.subs(sol_sub),t<=tc),(0.,True)))

        u_off = sp.integrate(g)
        u = u_off-u_off.subs(t,1)

        self._flow = sp.lambdify(t,u)
        self._flow_der = sp.lambdify(t,g)

        self.tp = tp
        self.te = te
        self.ta = ta
        self.tc = tc

    def set_shape_parameter(self,rd,alpha_guess=None):
        ra = (-1 + 4.8*rd)/100
        rk = (22.4 + 11.8*rd)/100
        rg = rk/4 / (.11*rd / (.5+1.2*rk)- ra)
        print ("ra = {}\nrk = {}\nrg = {}".format(ra,rk,rg))

        ta = ra
        tp = 1/2/rg
        te = rk*tp + tp

        print ("ta = {}\ntp = {}\nte = {}".format(ta,tp,te))

        if alpha_guess is None:
            if rd>=1:
                alpha_guess=2.
            else:
                alpha_guess=10.

        if alpha_guess is None:
            self.set_timing_parameters(ta=ta,te=te,tp=tp)
        else:
            self.set_timing_parameters(ta=ta,te=te,tp=tp,alpha_guess=alpha_guess)

    def __call__(self, t, der=0):
        tt = np.mod(t,1)
        if der == 0:
            return self._flow(tt)
        elif der == 1:
            return self._flow_der(tt)
        else:
            raise ValueError("Derivative must be 0 or 1")

    def fourier_components(self, nharm=100):
        npts = int(2**np.ceil(np.log2(nharm)))
        t = np.linspace(0,1,npts+1)
        x = self(t[:npts])
        xf = np.fft.fft(x)/npts
        exc = xf[:npts//2]
            
        exc[1:]*=2
        return exc

    def signal_gen(self, sr=16000, tf0=[0.,1.], f0=[150., 100.], amp=None):
        """
        generate an antialiased signal with constant glottal parameter
        and f0 given by tf0,f0
        """
        if amp is None:
            amp = np.ones(len(tf0))
        
        max_harm = sr/2/np.min(f0)
        exc = self.fourier_components(max_harm)

        t = np.arange(np.min(tf0),np.max(tf0),1./sr)
        tmax = max(tf0)
        f0t = np.interp(t,tf0,f0)
        a0t =  np.interp(t,np.linspace(0,tmax,len(amp)),amp)

        x = np.zeros(len(t))
        maxf = sr/2

        for nh in range(len(exc)):
            at = a0t.copy() 
            ft = f0t*nh
            cutidx = ft>maxf
            at[cutidx] = 0
            ph = np.cumsum(ft*2*np.pi/sr)
            x += np.real(at*exc[nh]*np.exp(1j*ph))

        return x

    def signal_gen_interpolator(self, sr=16000, tpar=[0,1], params=[{'f0':100,'f0':100}]):
        fourier = []
        f0 = []
        ampl = []
        last_f = 150
        last_a = 1.
        max_harm = 1
        for tt, pp in zip(tpar,params):
            try:
                last_f = pp.pop('f0')
            except KeyError:
                pass
            f0.append(last_f)
            
            try:
                last_a = pp.pop('ampl')
            except KeyError:
                pass
            ampl.append(last_a)

            self.set_timing_parameters(**pp)
            nharm = np.ceil(sr/last_f)
            max_harm = max(max_harm,nharm)
            fourier.append(self.fourier_components(nharm))

        
        t = np.arange(np.min(tpar),np.max(tpar),1./sr)
        tmax = max(tpar)
        f0t = np.interp(t,np.linspace(0,tmax,len(f0)),f0)

        for nh in range(max_harm):
            at = a0t.copy()
            ft = f0t*nh
            cutidx = ft>maxf
            at[cutidx] = 0
            ph = np.cumsum(ft*2*np.pi/sr)
            x += real(at*exc[nh]*exp(1j*ph))

        return x