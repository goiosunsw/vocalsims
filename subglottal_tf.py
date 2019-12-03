import numpy as np
import pympedance.Synthesiser as psyn

from json_object import JSONObject 
from scipy.optimize import minimize

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
        
    def set_glottis_from_json(self,json):
        self.glottis_channel_length = json['rectangular slot/length']
        self.glottis_channel_width = json['maximum aperture']
        self.glottis_channel_depth = json['rectangular slot/depth']
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
        vt_len = self.supraglottal.get_total_length()
        tmx = self.supraglottal.transfer_mx_at_freq(f,from_pos=0.0,to_pos=vt_len)
        tf_vt = tmx[1,1,:] + tmx[1,0,:]*z_vt
        if no_subglottal:
            return tf_vt
        else:
            z_sg = self.subglottal.get_input_impedance_at_freq(f)#/taz.char_impedance
        zg = self._calc_glottis_impedance(f)
        return zg/(zg+z_vt+z_sg) * tf_vt
        
