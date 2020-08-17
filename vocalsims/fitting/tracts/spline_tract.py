import os

import numpy as np
import pandas as pd
from scipy.signal import argrelmax

import pympedance.Synthesiser as psyn
from psopt import SplineOptimizer
from goios_utils.plotting import bodeplot

from scipy import interpolate


def vowel_geom_db(vowel, dbfile=None, db=None):
    """
    Read a tract geometry from dbfile, or from a pandas dataframe

    Returns a pympedance duct
    """
    
    if dbfile is not None:
        df=pd.read_csv(dbfile,index_col=0)
    else:
        df=db
    
    label = vowel
    raddf = np.sqrt(df[df.index.str.isnumeric().fillna(False)]/np.pi)*10
    raddf = raddf.set_index(raddf.index.astype('i'))

    deltadf = df.loc['delta']*10

    l = deltadf[label]/1000
    #x = (raddf.index.to_series()-1)*deltadf[label]
    r = raddf[label]/1000
    ac = psyn.AcousticWorld(temp=36.5,humid=1.0)
    duct = psyn.Duct(world=ac)
    for rr in r:
        if not np.isnan(rr):
            lastrad=rr
            duct.append_element(psyn.StraightDuct(length=l, radius=rr, loss_multiplier=5.0))
    duct.set_termination(psyn.FlangedPiston(radius=lastrad))


    return duct


def vowel_spec(duct, f, source_spec):

    tube = duct
    mouth_area = np.pi*tube.elements[-1].radius**2

    tmx = tube.transfer_mx_at_freq(f,from_pos=0.0,to_pos=tube.get_total_length())
    z = tube.get_input_impedance_at_freq(f)
    zc = tube.termination.get_characteristic_impedance()
    tfu = tmx[1,1,:] + tmx[1,0,:]*z
    tfp = tmx[0,1,:] + tmx[0,0,:]*z

    powf = np.abs(tfp*tfu*np.abs(source_spec)**2)
    radf = powf*f**2
    intf = radf*mouth_area
    return intf


def tube_to_logspec(x,r):
    l = np.diff(x)
    l0 = l[0]
    l[0] = l0/2
    l = np.concatenate((l,[l0/2]))
    tube = psyn.Duct()
    for rr, ll in zip(r,l):
        tube.append_element(psyn.StraightDuct(radius=rr,length=ll,loss_multiplier=5))
    tube.set_termination(psyn.FlangedPiston(radius=rr))
    spec = vowel_spec(tube,f,xf)
    return np.log10(np.abs(spec))

def obj_fun_spec(x,geom_func):
    r = geom_func(x)
    return tube_to_logspec(x,r)

class SplineTract(SplineOptimizer):
    def __init__(self, tube=None, default_radius=.01, default_length=0.15, *args, **kwargs):
        if tube is not None:
            # get the dimensions
            xm = np.cumsum([el.length for el in tube.elements])
            ym = np.asarray([el.radius for el in tube.elements])
            self.tube = tube
        try:
            xm = kwargs.pop('x')
        except KeyError:
            pass
        try:
            ym = kwargs.pop('y')
        except KeyError:
            pass

        try:
            self.tube
        except AttributeError:
            self.tube = self._x_to_tube()
            
        
        super().__init__(x=xm, y=ym,  *args, **kwargs)
        if 'xc' not in kwargs.keys():
            self.init_x = np.linspace(0, default_length, self.n_nodes)
            self.init_r = np.ones(self.n_nodes)*default_radius
        else:
            self.init_x = self.xc
            self.init_r = self.yc

        self.dx = 0.001
        
        for parname in self.params:
            if parname[0] == 'y':
                self.set_bounds(parname,(0.001,0.04))
        self.set_data(x=xm,y=ym)
        self.fit_spline()

    def _x_to_tube(self):
        tube = psyn.Duct()
        #xm = np.concatenate(([0],self.xm))
        l = np.diff(self.xm)
        l = np.concatenate(([l[0]/2,l[:-1],l[-1]/2]))
        for ll,rr in zip(l,self.ym):
            tube.append_element(psyn.StraightDuct(radius=rr,length=self.dx,loss_multiplier=5))
        term = self.tube.termination
        term.radius = rr
        tube.set_termination(term)

    def _spl_to_tube(self):
        tube = psyn.Duct()
        xmin = np.min(self.xm)
        xmax = np.max(self.xm)
        x = np.arange(xmin,xmax,self.dx)
        r = self.spl(x)

        for rr in r:
            tube.append_element(psyn.StraightDuct(radius=rr,length=self.dx,loss_multiplier=5))
        term = self.tube.termination
        term.radius = rr
        tube.set_termination(term)

    def rad_spec(self, f, source_spec):
        tube = self.tube
        mouth_area = np.pi*tube.elements[-1].radius**2

        tmx = tube.transfer_mx_at_freq(f,from_pos=0.0,to_pos=tube.get_total_length())
        z = tube.get_input_impedance_at_freq(f)
        zc = tube.termination.get_characteristic_impedance()
        tfu = tmx[1,1,:] + tmx[1,0,:]*z
        tfp = tmx[0,1,:] + tmx[0,0,:]*z

        powf = np.abs(tfp*tfu*np.abs(source_spec)**2)
        radf = powf*f**2
        intf = radf*mouth_area
        return intf
        
    def fpred(self, x, *parlist):
        self._param_list_to_spl(parlist)
        self._spl_to_tube()
        y = self.obj_fun_wrap(x, self.spl, parlist)
        return(y)
        
    def init_spec_params(self, fspec, spec, default_slope=12.):
        self.fspec=fspec
        self.spec=spec
        base_spec = self.rad_spec(self.fspec,self.fspec**(-default_slope/6))
        spec_diff = spec / base_spec
        #p = np.polyfit(1.67*np.log10(fspec),10*np.log10(spec_diff),1)  
        g = np.mean(10*np.log10(spec_diff))/2
        self.set_param('source_gain',g)
        self.set_param('source_slope',default_slope)
        print('=== Spectrum initialization ===')
        print('Source gain: ', self.param_dict['source_gain'], ' dB')
        print('Source slope: ', self.param_dict['source_slope'], ' dB/oct')
        self.set_constraints()

    def fit_to_spec(self, fspec, spec):
        """
        adjust the spline so that the transfer function 
        fits to the spectrum
        """
        self.obj_fun_old = self.obj_fun_wrap
        self.obj_fun_wrap = self.obj_fun_spec
        self.ym = spec
        self.fspec = fspec
        self.target_spec = spec
        #self.minimize_method = 'COBYLA'
        for parname in self.params:
            if parname[0] == '.' or parname[0] == '.':
                self.fitting_off(parname)
        self.set_constraints()
        #self.constraints=None
        newpars = self.optimize(spec,self.xm)
        return newpars

    def _calc_source_spec(self,parlist):
        source_slope = parlist[self._params.index('source_slope')]
        source_gain = parlist[self._params.index('source_gain')]
        #print(source_gain,source_slope)
        self.source_spec = self.fspec**(-source_slope/6.)*10**(source_gain/10)

    def obj_fun_spec(self, x, geom_fun, *extra):
        try:
            parlist = extra[0]
        except IndexError:
            try:
                parlist = self.fitted_params
            except (AttributeError, IndexError):
                parlist = self.vals
        self._calc_source_spec(parlist)
        rad_spec = self.rad_spec(self.fspec, self.source_spec)
        return rad_spec
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path, pyfile = os.path.split(__file__)
    dbfile = os.path.join(path,'Story1996_male01_cm2.csv')
    vowel = 'o'
    tube = vowel_geom_db(vowel,dbfile=dbfile)

    st = SplineTract(tube,n_nodes=13,degree=3)

    costs, xc, yc = st.run_multi_fit(iterations=5)

    fig,ax = plt.subplots(1)
    ax.plot(st.xm,st.ym)
    xp = np.linspace(np.min(st.xm),np.max(st.xm),200)
    for xci, yci in zip(xc,yc):
        st.set_control_points(xc=xci,yc=yci)
        lns=ax.plot(xci,yci,'.')
        ax.plot(xp,st.spl(xp),color=lns[0].get_color())

    plt.show()