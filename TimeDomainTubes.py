from collections import deque
import numpy as np
from scipy.special import struve, j1

class VTLoss(object):
    """
    Implements a viscothermal loss calculator 
    and filter approximation
    """
    def __init__(self,length=1.,radius=.01,loss_multiplier=1.0,speed_of_sound=345.):
        self.length = length
        self.radius = radius
        self.speed_of_sound = speed_of_sound
        self.gamma = 1.4
        self.rho = 1.2
        self.loss_multiplier=loss_multiplier
        self._intermediate_params()

    def _intermediate_params(self):
        viscous_boundary_layer_const = np.sqrt(2*np.pi/self.speed_of_sound/4e-8)
        thermal_boundary_layer_const = np.sqrt(2*np.pi/self.speed_of_sound/5.6e-8)
        self.rv_const = viscous_boundary_layer_const * self.radius
        self.rt_const = thermal_boundary_layer_const * self.radius

    def propagation_constant(self,f):
        """
        return frequency dependence of the propagation constant
        (this is k(\omega) in exp(1j*k(\omega)*l))
        """
        omega = 2*np.pi*f
        rv = self.rv_const * np.sqrt(f)
        rt = self.rt_const * np.sqrt(f)
        P = np.sqrt(2)/rv
        PQ = (self.gamma-1)*np.sqrt(2)/rt
        Zv = omega*self.rho*(P*(1.+3.*P/2.)+1j*(1.+P))
        Yt = omega/(self.rho*self.speed_of_sound**2) *\
                (PQ*(1.-PQ/(2*(self.gamma-1.)))+1j*(1.+PQ))

        # propagation constant
        return np.sqrt(Zv*Yt)

    def __call__(self,f):
        """
        returns the propagation losses:

        this is the ratio to a perfect wave propagation:

        L(f) = exp(1j*k(2*\pi*f)*l)/exp(1j*2*\pi*f/c*l)
        """
        G = self.propagation_constant(f)

        return np.exp((-G+1j*(2*np.pi*f/self.speed_of_sound))*self.length*self.loss_multiplier)


    def filter_approx(self, sr=1., n_poles=3, n_pts=None, fmin=None, fmax=None, weight_const=9., f=None):
        """
        Return a LTI filter approximation of the viscothermal losses

        Arguments:
        sr (float): sampling rate
        n_poles (int): Numer of poles for vectfit approximation
        n_pts (int): Number of frequency points at which to calculate the filter
        f_min (float): Minimum frequency for approximation
        f_max (maximum frequency for approximation)
        weight_const (float): weight constant for vectfit fitting in:
            weights = exp(-weight_const*f/fmax)

        (tested for n_poles = 3, sr=48000, fmin=1., npts=1024)
        """
        import vectfit_zd as vfzd 
        import scipy.signal as sig

        if n_pts is None:
            n_pts = 1024
        if fmin is None:
            fmin = sr/n_pts
        if f is None:
            f = np.linspace(fmin,sr/2,n_pts)
        loss = self(f)
        fmax = sr
        fidx = f<fmax
        #sr = 16000

        s = 1j*2*np.pi*f
        zvar = np.exp(s/sr)
        hf=loss

        fnorm = (f/max(f))
        weights = np.exp(-weight_const*fnorm)


        p,r,d,h=vfzd.vectfit_auto(hf[fidx],zvar[fidx],n_poles=3,weights=weights)
        hm = vfzd.model(zvar,p,r,d,h)

        b,a = sig.invresz(r,p,[d,h])
        poles = np.roots(a)
        zeros = np.roots(b)

        self.b = b
        self.a = a

        return RealTimeFilter(b=b,a=a)

class RadLoss(object):
    """
    Implements a radiation loss calculator 
    and filter approximation
    """
    def __init__(self,radius=.01,speed_of_sound=345.):
        self.radius = radius
        self.speed_of_sound = speed_of_sound
        self.gamma = 1.4
        self.rho = 1.2
        self.approx = False

    @property
    def char_impedance(self):
        c = self.speed_of_sound
        return self.rho*c/(self.radius**2*np.pi)
    
    def __call__(self,f):
        """
        returns the propagation losses:

        this is the ratio to a perfect wave propagation:

        L(f) = exp(1j*k(2*\pi*f)*l)/exp(1j*2*\pi*f/c*l)
        """
        c = self.speed_of_sound
        K = 2*np.pi*f/c
        ka = K * self.radius
        # not sure that Z0 should be the parent one...
        Z0 = self.char_impedance

        if self.approx:
            zfletch = (((ka)**2/2)**-1+1)**-1 + \
                       1j*((8*ka/3/np.pi)**-1 + (2/np.pi/ka)**-1)**-1
        else:
            zfletch = 1-j1(2*ka)/ka + 1j*struve(1,2*ka)/ka


        # Z_flange = Z0*zfletch

        return (zfletch-1)/(1+zfletch)

    def filter_approx(self, sr=1., n_poles=3, n_pts=None, 
                      fmin=None, fmax=None, weight_const=9., 
                      inc_real=False, f=None):
        """
        Return a LTI filter approximation of the viscothermal losses

        Arguments:
        sr (float): sampling rate
        n_poles (int): Numer of poles for vectfit approximation
        n_pts (int): Number of frequency points at which to calculate the filter
        f_min (float): Minimum frequency for approximation
        f_max (maximum frequency for approximation)
        weight_const (float): weight constant for vectfit fitting in:
            weights = exp(-weight_const*f/fmax)

        (tested for n_poles = 3, sr=48000, fmin=1., npts=1024)
        """
        import vectfit_zd as vfzd 
        import scipy.signal as sig

        if n_pts is None:
            n_pts = 1024
        if fmin is None:
            fmin = sr/n_pts
        if f is None:
            f = np.linspace(fmin,sr/2,n_pts)
        loss = self(f)
        fmax = sr
        fidx = f<fmax
        #sr = 16000

        s = 1j*2*np.pi*f
        zvar = np.exp(s/sr)
        hf=loss

        fnorm = (f/max(f))
        weights = np.exp(-weight_const*fnorm)


        p,r,d,h=vfzd.vectfit_auto(hf[fidx],zvar[fidx],n_poles=n_poles,weights=weights,inc_real=inc_real)
        hm = vfzd.model(zvar,p,r,d,h)

        b,a = sig.invresz(r,p,[d,h])
        poles = np.roots(a)
        zeros = np.roots(b)

        self.b = b
        self.a = a

        return RealTimeFilter(b=b,a=a)

def simple_relfection_filt(radius=0.01, speed_of_sound=345., sr=1.):
    """
    Create a filter with a very crude simplification of an open flanged reflection
    Arguments:
        radius (float): opening radius in m
        speed_of_sound (float)
        sr (float): sampling rate
    """
    import scipy.signal as sig
    f_cutoff = speed_of_sound/1.5/np.pi/radius
    if f_cutoff<sr:
        b,a = sig.butter(1,f_cutoff/sr,btype='high')
    else:
        b=np.array([0]); a=np.array([1])
    return RealTimeFilter(b=b-a,a=a)


class RealTimeFilter(object):
    """
    Implements a LTI filter to be used in real time,
    one sample at a time
    
    Filter is defined in the finite difference form:

    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]

    Create as:
        `filt = RealTimeFilter(b,a)`

        providing `b` and `a` as iterables

    Attributes
    ----------
    b: list with in-signal coefficients
    a: list with out-signal coefficients 

    Methods
    -------
 
    reset(): set delay lines to 0

    tick(x): insert new sample and output filtered sample

    set_coeffs(b,a): set coefficients without resetting dealy lines

    """

    def __init__(self,b=[1],a=[1]):
        """
        Create new filter with coefficients a and b

        `a` and `b` should be provided as lists
        """

        self.set_coeffs(b,a)
        self.init_delay_lines()
        self.counter = 0
    
    def init_delay_lines(self):
        self.in_buff = np.zeros(self.n*2)
        self.out_buff = np.zeros(self.n*2)

    def set_coeffs(self, b, a):
        self.a = a
        self.b = b 
        na = len(self.a)
        nb = len(self.b)
        self.n = max(na,nb)
        self.init_delay_lines()

    def increment_counter(self):
        self.counter += 1

    def tick(self, in_smpl):
        """
        Input a new sample and output one sample
        """
        self.increment_counter()
        idx = self.counter % self.n
        self.in_buff[idx] = self.in_buff[idx+self.n] = in_smpl
        in_buff = self.in_buff[idx+self.n:idx+self.n-len(self.b):-1]
        out_buff = self.out_buff[idx+self.n-1:idx+self.n-len(self.a):-1]
        out = np.sum(self.b*in_buff) - sum(self.a[1:]*out_buff)
        out /= self.a[0]
        self.out_buff[idx] = self.out_buff[idx+self.n] = out
        return out

    def next_output(self, in_smpl):
        """
        return next filter output without introducing a new input sample
        """
        idx = (self.counter + 1) % self.n
        in_buff = self.in_buff[idx+self.n-1:idx+self.n-len(self.b):-1]
        in_buff = np.concatenate(([in_smpl], in_buff))
        out_buff = self.out_buff[idx+self.n-1:idx+self.n-len(self.a):-1]
        #import pdb
        #pdb.set_trace()
        out = np.sum(self.b*in_buff) - sum(self.a[1:]*out_buff)
        out /= self.a[0]
        return out


    def reset(self, in_val=0, out_val=0):
        """
        Resets the circular buffers to 0
        """
        self.init_delay_lines()
        self.counter = 0

class IntegerDelayLine(object):
    """
    Implements a delay line for use in real time
    
    Create as:
        `dl = IntegerDelayLine(delay)`
        delay: (int)


    Methods
    -------
 
    reset(): set delay line to 0

    tick(x): insert new sample and output filtered sample

    """

    def __init__(self, delay=1):
        """
        Create new delay with an integer delay del 
        """

        self.n = delay+1
        self.delay = delay
        self.init_delay_line()
        self.counter = 0
    
    def init_delay_line(self):
        self.buff = np.zeros(self.n*2)

    def increment_counter(self):
        self.counter += 1

    def tick(self, in_smpl):
        """
        Input a new sample and output one sample
        """

        self.increment_counter()
        idx = self.counter % self.n
        self.buff[idx] = self.buff[idx+self.n] = in_smpl
        out = self.buff[idx+self.n-self.delay]
        return out

    def next_output(self):
        """
        return next filter output without introducing a new input sample
        """
        idx = (self.counter + 1) % self.n
        out = self.buff[idx+self.n-self.delay]
        return out

    def get_line(self):
        idx = self.counter % self.n
        return self.buff[idx+self.n:idx+self.n-self.delay:-1]

    def reset(self, in_val=0, out_val=0):
        """
        Resets the circular buffers to 0
        """
        self.init_delay_line()
        self.counter = 0

    def get_delay(self):
        return self.n-1


class PerfectTube(object):
    """
    A tube with discrete length and no losses

    (length = 1 coresponds to an actual length of 1.5
     due to an additional delay at the reflection/transmission stage)
    """

    def __init__(self, length=1):
        self.dlin = IntegerDelayLine(length)
        self.dlout = IntegerDelayLine(length)

    def insert_outgoing(self, in_smpl):
        """
        Insert a sample in the outgoing line and
        return sample at far end
        """
        return self.dlout.tick(in_smpl)

    def insert_incoming(self, in_smpl):
        """
        Insert a sample in the incomin line and 
        return sample at near end
        """
        return self.dlin.tick(in_smpl)

    def dump_delays(self):
        """
        dumps the contents of both delays
        incoming delay is dumped in causal order
        outgoing delay is dumped in reverse order
        so that they both correspond to same geometrical locations
        """
        incoming = np.flipud(self.dlin.get_line())
        outgoing = (self.dlout.get_line())
        return (outgoing,incoming)

    def read_next_out_without_tick(self):
        """
        get the next outgoing sample without advancing the delay line
        """
        return self.dlout.get_line()[-1]

    def read_next_in_without_tick(self):
        """
        get the next incoming sample without advancing the delay line
        (same as the next output of tick())
        """
        return self.dlin.get_line()[-1]

    def reset(self):
        self.dlin.reset()
        self.dlout.reset()

class ConstFreqLossTube(PerfectTube):
    def __init__(self, length=1,losses=0.0):
        super().__init__(length=length)
        self.losses = losses    
        self.prop_mult = (1-losses)**length
        self.smpl_prop_mult = 1-losses
    
    def insert_incoming(self,smpl):
        out = super().insert_incoming(smpl)
        return out*self.prop_mult

    def insert_outgoing(self,smpl):
        out = super().insert_outgoing(smpl)
        return out*self.prop_mult

    def read_next_in_without_tick(self):
        out = super().read_next_in_without_tick()
        return out*self.prop_mult
    
    def read_next_out_without_tick(self):
        out = super().read_next_out_without_tick()
        return out*self.prop_mult


class LossyTube(PerfectTube):
    def __init__(self, length=1):
        super().__init__(length=length)
        self.prop_out_obj = RealTimeFilter([1],[1])
        self.prop_in_obj = RealTimeFilter([1],[1])

    def set_propagation_filter(self,b,a):
        self.prop_in_obj = RealTimeFilter(b,a)
        self.prop_out_obj = RealTimeFilter(b,a)

    def set_lossless_propagation(self):
        self.prop_in_obj = RealTimeFilter([1],[1])
        self.prop_out_obj = RealTimeFilter([1],[1])

    def insert_incoming(self,smpl):
        out = super().insert_incoming(smpl)
        return self.prop_in_obj.tick(out)

    def insert_outgoing(self,smpl):
        out = super().insert_outgoing(smpl)
        return self.prop_out_obj.tick(out)

    def read_next_in_without_tick(self):
        out = super().read_next_in_without_tick()
        return self.prop_in_obj.next_output(out)
    
    def read_next_out_without_tick(self):
        out = super().read_next_out_without_tick()
        return self.prop_out_obj.next_output(out)

    def reset(self):
        try:
            self.prop_in_obj.reset()
        except AttributeError:
            pass


class ViscoThermalTube(LossyTube):
    def __init__(self, length=1., radius=.1,
                 loss_multiplier=1.,
                 sr=48000, speed_of_sound=345.,
                 n_poles=3, n_pts=None, fmin=None, fmax=None, f=None,
                 weight_constant=9., lossy=True):

        self.requested_length = length
        # length corresponding to delay line
        smpl_length = speed_of_sound/sr
        n_smpl_all = length/smpl_length
        # half a sample is allocated for reflection or
        # transmission to next tube
        n_smpl_prop = int(n_smpl_all-.5)
        length = n_smpl_prop*smpl_length
        self.physical_length = length
        self.physical_radius = radius
        self.loss_multiplier = loss_multiplier
        self.speed_of_sound = speed_of_sound
        self.sr = sr

        super().__init__(length=n_smpl_prop)
        if lossy:
            if n_pts is None:
                n_pts = 1024
            f = np.linspace(1,sr/2,n_pts)
            self._make_lossy(f=f)
        else:
            self._make_lossless()

    def _make_lossy(self, f):
        vtl = VTLoss(length=self.physical_length,
                     radius=self.physical_radius,
                     loss_multiplier=self.loss_multiplier,
                     speed_of_sound=self.speed_of_sound)

        vtlf = vtl.filter_approx(f=f,sr=self.sr)
        b, a = vtlf.b, vtlf.a
        self.set_propagation_filter(b,a)

    def _make_lossless(self):
        self.set_lossless_propagation()


class SimpleTubeWithTermination(object):
    """
    A single straight tube segment associated 
    with a reflection condition 
    """
    def __init__(self, length=1, reflection_coeff=1):
        self.tube = PerfectTube(length=length)
        if type(reflection_coeff) is RealTimeFilter:
            self.rfunc = reflection_coeff.tick
        else:
            self.rfunc = lambda x: x*reflection_coeff

    def tick(self, in_smpl):
        out_last = self.tube.insert_outgoing(in_smpl)
        in_first = self.tube.insert_incoming(self.rfunc(out_last))
        return in_first

    def dump_delays(self):
        return self.tube.dump_delays()

class SingleElementLossyTube(SimpleTubeWithTermination):
    def __init__(self, length=1, reflection_coeff=1):
        self.tube = LossyTube(length=length)
        if type(reflection_coeff) is RealTimeFilter:
            self.rfunc = reflection_coeff.tick
        else:
            self.rfunc = lambda x: x*reflection_coeff


class RealTimeDuct(object):
    """
    A tube assembly with straight elements of varying 
    dimensions.

    create with:
        ta = self.TubeAssembly(reflection_coeff=1)
    
    append elements with:
        ta.append_tube(length=.17,radius=.02)

    send a new outgoing pressure sample with:
        p_in = ta.tick(p_out_smpl)
        (p_in is the reflected pulse)
    """
    def __init__(self, open=True, lossy=True, 
                 speed_of_sound=345., sr=48000,
                 loss_multiplier=1.0,
                 simpl_reflection=True):
        self.tubes = []
        if open:
            reflection_coeff = -1
        else:
            reflection_coeff = 1
        self.loss_multiplier = loss_multiplier
        self.rfunc = lambda x: x*reflection_coeff
        self.reflection_coeff = reflection_coeff
        #self.radii = []
        self.reflection_coeffs = []
        self.transmission_coeffs = []
        self.default_mx = np.array([[1.,0.],[0.,1.]])
        self.scats = [np.array([[0.,1.],[1.,0.]])]
        self.lossy = lossy
        self.sr = sr
        self.speed_of_sound = speed_of_sound
        self.end_out_last = 0.
        self.end_in_last = 0.
        self.simpl_reflection = simpl_reflection

    @property
    def radii(self):
        return [xx.physical_radius for xx in self.tubes]

    def append_tube(self, length=1, radius=1, loss_multiplier=None):
        try:
            prev_radius = self.tubes[-1].physical_radius
            zrat = (radius/prev_radius)**2
            #self.reflection_coeffs.insert(-1,.5*(1-zrat))
            #self.transmission_coeffs.insert(-1,.5*(1+zrat))
        except IndexError:
            pass
            #self.reflection_coeffs.append(self.reflection_coeff)
            #self.transmission_coeffs.append(1-self.reflection_coeff)

        if loss_multiplier is None:
            loss_multiplier = self.loss_multiplier

        new_tube = ViscoThermalTube(length=length,radius=radius,
                                     speed_of_sound=self.speed_of_sound,
                                     sr=self.sr, loss_multiplier=loss_multiplier)
        self.tubes.append(new_tube)        
        self.scats.append(self.default_mx)
        self.connect_tubes()

    def adjust_radius(self, radius, index=-1):
        self.tubes[index].physical_radius = radius
        self.connect_tubes(index)
        if index == -1:
            self.adjust_termination()

    def connect_tubes(self,index=-1):
        if index<0:
            index = len(self.tubes)+index
        if index==0:
            return
        # scalers (proportional to zc)
        scr = 1/self.tubes[index].physical_radius**2
        scl = 1/self.tubes[index-1].physical_radius**2
        newscat = np.array([[2*scr, scl-scr],[scr-scl, 2*scl]])/(scr+scl)
        self.scats[index] = newscat

    def iter_tubes(self,reverse=False):
        """
        iterate through tube elements in the assembly

        (returns one tube per iteration)
        """
        if reverse:
            idx = range(len(self.tubes)-1,0,-1)
        else:
            idx = range(len(self.tubes))
        for ii in idx:
            yield self.tubes[ii].physical_radius,self.tubes[ii]

    def iter_junctions(self, reverse=False, edges=True):
        """
        iterate through tube junctions in the assembly

        (returns tubes on either side and scattering matrix)
        """
        if edges:
            first_ind = 1
            last_ind = len(self.tubes)+2
            tubes = ([None])
            tubes.extend(self.tubes)
            tubes.extend([None])
            scatoff=-1
        else:
            first_ind = 1
            last_ind = len(self.tubes)
            tubes = self.tubes
            scatoff=0

        if reverse:
            idx = range(last_ind-1,first_ind-1,-1)
        else:
            idx = range(first_ind,last_ind)

        for ii in idx:
            if reverse:
                inext = ii-1
                iprev = ii
            else:
                inext = ii
                iprev = ii-1
            yield (tubes[iprev], 
                   tubes[inext], 
                   self.scats[ii+scatoff])

    def adjust_termination(self):
        last_rad = self.tubes[-1].physical_radius
        if self.lossy:
            if self.simpl_reflection:
                self.rlfilt = simple_relfection_filt(radius=last_rad,
                                                speed_of_sound=self.speed_of_sound,
                                                sr=self.sr)
            else:
                rl = RadLoss(radius=last_rad,
                             speed_of_sound=self.speed_of_sound)
                self.rlfilt = rl.filter_approx(sr=self.sr)
            self.rfunc = self.rlfilt.tick


    def tick(self, in_smpl):
        out_next = in_smpl
        # samples leaving junctions 
        # [to the outgoing line on the next tube,
        #  to the incoming line on the previous tube]
        all_leaving = [[in_smpl,0.]]
        tube_next = self.tubes[0]
        # run through the tube and calculate incoming 
        # values for each junction without ticking
        for tube, tube_next, scat in self.iter_junctions(edges=False):
            # incoming wave from next tube
            in_next = tube_next.read_next_in_without_tick()
            # outgoing wave from previous tube
            out_prev = tube.read_next_out_without_tick()
            arriving = np.array([out_prev, in_next])
            leaving = np.dot(scat,arriving)
            all_leaving.append(leaving)

        # calc last tube and get sample to reflect
        tube = tube_next
        scat = self.scats[-1]
        #arriving = np.array([tube.read_next_out_without_tick(),
        #                    0.])
        #leaving = np.dot(scat,arriving)
        out_last = tube.read_next_out_without_tick()
        #out_last = tube.insert_outgoing(in_smpl)
        reflected = (self.rfunc(out_last))
        all_leaving.append([0.,reflected])
        #import pdb
        #pdb.set_trace()
        self.end_out_last = out_last 
        self.end_in_last = reflected

        # insert "leaving" values
        for leaving, (tube, tube_next, scats) in zip(all_leaving[::-1],
                                                  self.iter_junctions(edges=True,
                                                                      reverse=True)):
            # incoming wave of this tube
            if tube:
                in_l=tube.insert_outgoing(leaving[0])
            if tube_next:
                out_l=tube_next.insert_incoming(leaving[1])

        return out_l
    
    def read_next_in_without_tick(self):
        tube = self.tubes[0]
        out = tube.read_next_in_without_tick()
        return tube.prop_in_obj.next_output(out)
    
    def read_next_out_without_tick(self):
        tube = self.tubes[-1]
        out = tube.read_next_out_without_tick()
        return tube.prop_out_obj.next_output(out)

    def dump_delays(self):
        ins = []
        outs = []
        for radius, tube in self.iter_tubes():
            (this_out,this_in) = tube.dump_delays()
            ins.extend(this_in)
            outs.extend(this_out)
        return(outs,ins)

    def to_physical_dimensions(self, speed_of_sound=345., sampling_rate=48000):
        radii = []
        lengths = [] 
        pre = 0
        for rr, tube in self.iter_tubes():
            ll = (tube.dlin.get_delay())
            radii.append(rr)
            lengths.append((ll+pre+.5)*speed_of_sound/sampling_rate)
            pre=.5 
        return radii, lengths

    def reset(self):
        for tube in self.tubes:
            tube.reset()
        try:
            self.rlfilt.reset()
        except AttributeError:
            print("No radiation filter")
            
    def impulse_response(self,n=1024):
        self.reset()
        y=[]
        y.append(self.tick(1.))
        for ii in range(n):
            y.append(self.tick(0.))

        return np.array(y)

    def transfer_impulse_response(self,n=1024):
        self.reset()
        y = []
        to = []
        ti = []
        y.append(self.tick(1.))
        to.append(self.end_out_last)
        ti.append(self.end_in_last)
        for ii in range(n):
            y.append(self.tick(0.))
            to.append(self.end_out_last)
            ti.append(self.end_in_last)

        return np.array(to), np.array(ti), np.array(y)

    def impulse_map(self,n=1024):
        self.reset()
        ddout=[]
        ddin=[]
        out = (self.tick(1.))
        ddo,ddi = self.dump_delays()
        ddout.append(ddo)
        ddin.append(ddi)
        for ii in range(n):
            out = (self.tick(0.))
            ddo,ddi = self.dump_delays()
            ddout.append(ddo)
            ddin.append(ddi)

        return np.array(ddout),np.array(ddin)

        

class TubeAssembly(object):
    """
    A tube assembly with straight elements of varying 
    dimensions.

    create with:
        ta = self.TubeAssembly(reflection_coeff=1)
    
    append elements with:
        ta.append_tube(length=.17,radius=.02)

    send a new outgoing pressure sample with:
        p_in = ta.tick(p_out_smpl)
        (p_in is the reflected pulse)
    """
    def __init__(self, reflection_coeff=1, lossy=True):
        self.tubes = []
        if type(reflection_coeff) is RealTimeFilter:
            self.rfunc = reflection_coeff.tick
        else:
            self.rfunc = lambda x: x*reflection_coeff
        self.reflection_coeff = reflection_coeff
        self.radii = []
        self.reflection_coeffs = []
        self.transmission_coeffs = []
        self.end_mx = np.array([[0.,1.],[1.,0.]])
        self.scats = [np.array([[0.,1.],[1.,0.]])]
        self.lossy = lossy
        self.end_out_last = 0.
        self.end_in_last = 0.

    def append_tube(self, length=1, radius=1):
        try:
            prev_radius = self.radii[-1]
            zrat = (radius/prev_radius)**2
            #self.reflection_coeffs.insert(-1,.5*(1-zrat))
            #self.transmission_coeffs.insert(-1,.5*(1+zrat))
        except IndexError:
            pass
            #self.reflection_coeffs.append(self.reflection_coeff)
            #self.transmission_coeffs.append(1-self.reflection_coeff)

        self.radii.append(radius)
        if self.lossy:
            new_tube = LossyTube(length)
        else:
            new_tube = PerfectTube(length)
        self.tubes.append(new_tube)        
        self.scats.append(self.end_mx)
        self.connect_tubes()

    def connect_tubes(self,index=-1):
        if index<0:
            index = len(self.tubes)+index
        if index==0:
            return
        # scalers (proportional to zc)
        scr = 1/self.radii[index]**2
        scl = 1/self.radii[index-1]**2
        newscat = np.array([[2*scr, scl-scr],[scr-scl, 2*scl]])/(scr+scl)
        self.scats[index] = newscat

    def iter_tubes(self,reverse=False):
        """
        iterate through tube elements in the assembly

        (returns one tube per iteration)
        """
        if reverse:
            idx = range(len(self.tubes)-1,0,-1)
        else:
            idx = range(len(self.tubes))
        for ii in idx:
            yield self.radii[ii],self.tubes[ii]

    def iter_junctions(self, reverse=False, edges=True):
        """
        iterate through tube junctions in the assembly

        (returns tubes on either side and scattering matrix)
        """
        if edges:
            first_ind = 1
            last_ind = len(self.tubes)+2
            tubes = ([None])
            tubes.extend(self.tubes)
            tubes.extend([None])
            scatoff=-1
        else:
            first_ind = 1
            last_ind = len(self.tubes)
            tubes = self.tubes
            scatoff=0

        if reverse:
            idx = range(last_ind-1,first_ind-1,-1)
        else:
            idx = range(first_ind,last_ind)

        for ii in idx:
            if reverse:
                inext = ii-1
                iprev = ii
            else:
                inext = ii
                iprev = ii-1
            yield (tubes[iprev], 
                   tubes[inext], 
                   self.scats[ii+scatoff])


    def tick(self, in_smpl):
        out_next = in_smpl
        # samples leaving junctions 
        # [to the outgoing line on the next tube,
        #  to the incoming line on the previous tube]
        all_leaving = [[in_smpl,0.]]
        # run through the tube and calculate incoming 
        # values for each junction without ticking
        tube_next = self.tubes[0]

        for tube, tube_next, scat in self.iter_junctions(edges=False):
            # incoming wave from next tube
            in_next = tube_next.read_next_in_without_tick()
            # outgoing wave from previous tube
            out_prev = tube.read_next_out_without_tick()
            arriving = np.array([out_prev, in_next])
            leaving = np.dot(scat,arriving)
            all_leaving.append(leaving)

        # calc last tube and get sample to reflect
        tube = tube_next
        scat = self.scats[-1]
        #arriving = np.array([tube.read_next_out_without_tick(),
        #                    0.])
        #leaving = np.dot(scat,arriving)
        out_last = tube.read_next_out_without_tick()
        #out_last = tube.insert_outgoing(in_smpl)
        reflected = (self.rfunc(out_last))
        all_leaving.append([0.,reflected])
        #import pdb
        #pdb.set_trace() 
        self.end_out_last = out_last 
        self.end_in_last = reflected


        # insert "leaving" values
        for leaving, (tube, tube_next, scats) in zip(all_leaving[::-1],
                                                  self.iter_junctions(edges=True,
                                                                      reverse=True)):
            # incoming wave of this tube
            if tube:
                in_l=tube.insert_outgoing(leaving[0])
            if tube_next:
                out_l=tube_next.insert_incoming(leaving[1])

        return out_l

    def dump_delays(self):
        ins = []
        outs = []
        for radius, tube in self.iter_tubes():
            (this_out,this_in) = tube.dump_delays()
            ins.extend(this_in)
            outs.extend(this_out)
        return(outs,ins)

    def to_physical_dimensions(self, speed_of_sound=345., sampling_rate=48000):
        radii = []
        lengths = [] 
        pre = 0
        for rr, tube in self.iter_tubes():
            ll = (tube.dlin.get_delay())
            radii.append(rr)
            lengths.append((ll+pre+.5)*speed_of_sound/sampling_rate)
            pre=.5 
        return radii, lengths

    def read_next_in_without_tick(self):
        tube = self.tubes[0]
        out = tube.read_next_in_without_tick()
        return tube.prop_in_obj.next_output(out)
    
    def read_next_out_without_tick(self):
        tube = self.tubes[-1]
        out = tube.read_next_out_without_tick()
        return tube.prop_out_obj.next_output(out)

    def to_physical_dimensions(self, speed_of_sound=345., sampling_rate=48000):
        radii = []
        lengths = [] 
        pre = 0
        for rr, tube in self.iter_tubes():
            ll = (tube.dlin.get_delay())
            radii.append(rr)
            lengths.append((ll+pre+.5)*speed_of_sound/sampling_rate)
            pre=.5 
        return radii, lengths

    def reset(self):
        for tube in self.tubes:
            tube.reset()
            
    def impulse_response(self,n=1024):
        self.reset()
        y=[]
        y.append(self.tick(1.))
        for ii in range(n):
            y.append(self.tick(0.))

        return np.array(y)

    def transfer_impulse_response(self,n=1024):
        self.reset()
        y = []
        to = []
        ti = []
        y.append(self.tick(1.))
        to.append(self.end_out_last)
        ti.append(self.end_in_last)
        for ii in range(n):
            y.append(self.tick(0.))
            to.append(self.end_out_last)
            ti.append(self.end_in_last)

        return np.array(to), np.array(ti), np.array(y)

    def impulse_map(self,n=1024):
        self.reset()
        ddout=[]
        ddin=[]
        out = (self.tick(1.))
        ddo,ddi = self.dump_delays()
        ddout.append(ddo)
        ddin.append(ddi)
        for ii in range(n):
            out = (self.tick(0.))
            ddo,ddi = self.dump_delays()
            ddout.append(ddo)
            ddin.append(ddi)

        return np.array(ddout),np.array(ddin)

class ConstFreqLossDuct(TubeAssembly):
    """
    A wrapper around a tube assembly using physical 
    dimensions
    """
    def __init__(self, reflection_coeff=1, lossy=True,
                 speed_of_sound=345., sr=48000):
        self.sr = sr
        self.speed_of_sound = speed_of_sound
        super().__init__(reflection_coeff=reflection_coeff,
                         lossy=lossy)
    
    def append_tube(self, length=1., radius=.01):
        requested_length = length
        # length corresponding to delay line
        smpl_length = self.speed_of_sound/self.sr
        n_smpl_all = length/smpl_length
        # half a sample is allocated for reflection or
        # transmission to next tube
        n_smpl_prop = int(n_smpl_all-.5)
        length = n_smpl_prop*smpl_length
        physical_length = length
        physical_radius = radius
        try:
            prev_radius = self.radii[-1]
            zrat = (radius/prev_radius)**2
            #self.reflection_coeffs.insert(-1,.5*(1-zrat))
            #self.transmission_coeffs.insert(-1,.5*(1+zrat))
        except IndexError:
            pass
            #self.reflection_coeffs.append(self.reflection_coeff)
            #self.transmission_coeffs.append(1-self.reflection_coeff)

        self.radii.append(radius)
        if self.lossy:
            new_tube = LossyTube(length)
        else:
            new_tube = PerfectTube(length)
        self.tubes.append(new_tube)        
        self.scats.append(self.end_mx)
        self.connect_tubes()

    def adjust_termination(self):
        pass

class Tube(object):
    def __init__(self, delay=1, losses=0.):
        self.extra=5
        self.dlin = DelayLine(delay+1,extra=self.extra)
        self.dlout = DelayLine(delay+1,extra=self.extra)
        self.scat = None
        self.tube_l = None
        self.tube_r = None
        self.delay = delay
        self.prop_mult = (1-losses)**delay
        self.smpl_prop_mult = 1-losses
    def insert_incoming_sample(self,sin):
        self.dlin.insert_sample(sin)
    def insert_outgoing_sample(self,sin):
        self.dlout.insert_sample(sin)
    def read_outgoing(self):
        return self.dlout.read_delay(self.dlout.delay-2)*self.prop_mult
    def read_incoming(self):
        return self.dlin.read_delay(self.dlin.delay-2)*self.prop_mult
    def read_samples(self):
        return self.dlout.read_delay(), self.dlin.read_delay()
    
    def read_incoming_at_pos(self,index):
        return self.dlin.read_delay(self.delay-index)*(self.smpl_prop_mult)**(self.delay-index)
    def read_outgoing_at_pos(self,index):
        return self.dlout.read_delay(index)*(self.smpl_prop_mult)**(index)

    def get_sum_at_pos(self,index):
        #pini = self.dlin.read_delay(self.delay-index)
        #pouti = self.dlout.read_delay(index)
        pini = self.read_incoming_at_pos(index)
        pouti = self.read_outgoing_at_pos(index)
        return pini + pouti
    def get_diff_at_pos(self,index):
        pini = self.dlin.read_delay(self.delay-index)
        pouti = self.dlout.read_delay(index)
        return -pini + pouti
    def get_sum_distribution(self):
        pmults = self.smpl_prop_mult**np.arange(self.delay+1)
        pout = np.array(self.dlout.dump_line())*pmults
        pin = np.flipud(np.array(self.dlin.dump_line())*pmults)
        return pout+pin
        