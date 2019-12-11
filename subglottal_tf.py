import numpy as np
import pympedance.Synthesiser as psyn
from datetime import datetime
import argparse 
from hdf5_interface import Hdf5Interface

from json_object import JSONObject 
from scipy.optimize import minimize
import json_iterators as jsi

__simulation_type__ = "freq-domain simulation with subglottal tract (Hanson & Stevens 1996)"
__simulation_version__ = "20191128"
__parameter_version__ = "20191128"

class CustomAcousticWorld(psyn.AcousticWorld):
    def __init__(self, json=None):
        super().__init__()
        if json is not None:
            self.from_json(json)

    def from_json(self, json):
        try: 
            jac = json['acoustic']
            assert jac['_prefer']
        except (KeyError, AssertionError):
            print("Setting based on physical parameters")
            jph = json['physical']
            self.temperature = jph['temperature']
            self.humidity = jph['humidity']
            self.pressure = jph['atmospheric pressure']
        else:
            print("Setting based on acoustic parameters")
            self.speed_of_sound = jac['speed of sound']
            self.viscosity = jac['viscosity']
            self.medium_density = jac['density']

class FrequencyDomainTract(psyn.Duct):
    def __init__(self, json=None, world=None):
        super(FrequencyDomainTract,self).__init__(world=world)
        if json is not None:
            self.from_json(json)

    
    def from_json(self, json):
        try:
            self.description = json['description']
        except KeyError:
            self.description = "Unnamed tract"
        for ii,el in json['elements'].items():
            try:
                l = el['length']
                r = el['radius']
            except KeyError:
                pass
            if el['type'] == 'cylinder':
                l = el['length']
                r = el['radius']
                try:
                    mult = el['loss multiplier']
                    self.append_element(psyn.StraightDuct(l,r,loss_multiplier=mult))
                except KeyError:
                    self.append_element(psyn.StraightDuct(l,r))
                    
            elif el['type'] == 'termination':
                if el['kind'] == 'open':
                    self.set_termination(psyn.FlangedPiston(radius=r))
                elif el['kind'] == 'closed':
                    self.set_termination(psyn.PerfectClosedEnd())
            else:
                print("Not Implemented: {}".format(el['type']))

class FreqDomainVocalSim(object):
    def __init__(self, json=None):
        super().__init__()
        self.tracts = {}
        if json is not None:
            self.from_json(json)    

    def reset(self):
        pass
            
    def set_hdf5_path(self,file,path):
        self.hdf5 = Hdf5Interface(file,path)

    def from_json(self, json):
        self.json = json
        world = CustomAcousticWorld(json['environment'])
        self.viscosity = json['environment/acoustic/viscosity']
        for k, v in json['tracts'].items():
            self.tracts[k] = FrequencyDomainTract(json=v,world=world)
        self.supraglottal = self.tracts['supraglottal']
        self.subglottal = self.tracts['subglottal']
        self.set_glottis_from_json(json['glottis'])
        self.p_lung = json['environment/lung pressure']

    def to_hdf5(self):
        self.hdf5.write_attrs({"json":self.json.dumps()})
        self.hdf5.write_dataset(self.tf_vt,name="transfer function/supraglottal")
        self.hdf5.write_dataset(self.tf,name="transfer function/Hanson Stevens 1996")
        
    def set_glottis_from_json(self,json):
        self.glottis_channel_length = json['rectangular slot/length']
        self.glottis_channel_depth = json['rectangular slot/depth']
        max_a = json['maximum aperture']
        self.glottis_channel_width = max_a/self.glottis_channel_length
        self.Krho = json['rectangular slot/krho']
        
    def _calc_glottis_impedance(self, f):
        rho = self.supraglottal.world.medium_density

        mu = self.viscosity
        lg = self.glottis_channel_length
        dg = self.glottis_channel_width
        hg = self.glottis_channel_depth
        
        # average flow through glottis
        Ug = np.sqrt(2*self.p_lung/rho)*dg*lg
        # resistance of glottis channel
        rg = 12*mu*hg/lg/dg**3 + self.Krho*Ug/(lg*dg)**2
        # acoustic mass of glottis channel
        mg = rho*hg/lg/dg
        
        return rg+1j*2*np.pi*f*mg

    def supraglottal_impedance_from_glottis(self,f):
        return self.supraglottal.get_input_impedance_at_freq(f)
    def subglottal_impedance_from_glottis(self,f):
        return self.subglottal.get_input_impedance_at_freq(f)

    def transfer_function_flow_flow(self,f,no_subglottal=False):
        z_vt = self.supraglottal.get_input_impedance_at_freq(f)#/taz.char_impedance
        self.z_vt = z_vt
        vt_len = self.supraglottal.get_total_length()
        tmx = self.supraglottal.transfer_mx_at_freq(f,from_pos=0.0,to_pos=vt_len)
        tf_vt = tmx[1,1,:] + tmx[1,0,:]*z_vt
        if no_subglottal:
            return tf_vt
            #self.z_sg = None
        else:
            z_sg = self.subglottal.get_input_impedance_at_freq(f)#/taz.char_impedance
            self.z_sg = z_sg
        zg = self._calc_glottis_impedance(f)
        self.zg = zg
        return zg/(zg+z_vt+z_sg) * tf_vt

    def simulate(self):
        fstart = self.json['frequency domain/start']
        fstop = self.json['frequency domain/stop']
        n = self.json['frequency domain/n']
        fvec = np.linspace(fstart,fstop,n)
        self.tf = self.transfer_function_flow_flow(fvec)
        self.tf_vt = self.transfer_function_flow_flow(fvec,no_subglottal=True)
        try:
            self.to_hdf5()
        except AttributeError:
            pass

    def plots(self):
        import matplotlib.pyplot as pl
        fig,ax = pl.subplots(2,2,sharex=True)
        
        def bodeplot(zz,ax, label=""):
            lns = ax[0].semilogy(np.abs(zz),label="")
            ax[1].plot(np.angle(zz),label="",color=lns[0].get_color())
        bodeplot(self.z_vt,ax = ax[:,0],label="vt")
        bodeplot(self.z_sg,ax = ax[:,0],label="sg")
        bodeplot(self.zg,ax = ax[:,0],label="glot")

        bodeplot(self.tf,ax = ax[:,1],label="HS")
        bodeplot(self.tf_vt,ax = ax[:,1],label="vt")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file',help='input file with parameters in json format')
    parser.add_argument('-p','--plot',action='store_true', help='produce plot when finished')
    parser.add_argument('-o','--output',help='output HDF5 file')
    args = parser.parse_args()

    infile = args.param_file

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not args.output:
        args.output = timestamp+'.hdf5'

    output = args.output

    sim = FreqDomainVocalSim()
    with open(infile) as f:
        js = JSONObject(f)
    try:
        jseq = js['sequence']
        print('Sequence found')
    except KeyError:
        jseq = None
    
    if jseq:
        for jj in jsi.json_expander(js, jseq, ignore_kw="_ignore_freq_domain"):
            print(jj)
            sim.reset()
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
            sim.set_hdf5_path(output,timestamp)
            sim.from_json(jj)
            sim.simulate()


    if args.plot:
        from matplotlib.pyplot import plot, show
        p_sg = sim.p_sg_in + sim.p_sg_out;
        p_vt = sim.p_vt_in + sim.p_vt_out;

        u = (sim.p_vt_out - sim.p_vt_in)/sim.zc_vt;
        u_sg = -(sim.p_sg_out - sim.p_sg_in)/sim.zc_sg

        plot(u)
        show()

