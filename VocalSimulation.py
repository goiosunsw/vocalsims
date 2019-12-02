import TimeDomainTubes as tdt




        
class CompleteVocalTract(object):
    def __init__(self,subglottal_lengths=None,
                      subglottal_radii=0.01,
                      supraglottal_lengths=.17,
                      supraglottal_radii=.02):
        # tract objects
        self.supraglottal = None
        self.subglottal = None
        
        # glottis parameters in m
        # approx length of folds that is separated
        self.glottis_channel_length = 0.015
        # average distance between folds
        self.glottis_channel_width = 0.001
        # streamwise
        self.glottis_channel_depth = 0.003
        
        # kinematic viscosity (mu)
        self.viscosity = 18.84e-6
        # 
        self.Krho=1.
        
        # lung pressure in Pa
        self.p_lung = 700
        
        # default termination for supraglottal
        self._supraglottal_term = psyn.FlangedPiston
        self._subglottal_term = psyn.FlangedPiston
        
        if subglottal_lengths is not None:
            try: 
                len(subglottal_lengths)
            except TypeError:
                subglottal_lengths = [subglottal_lengths]
            try:
                len(subglottal_radii)
            except TypeError:
                subglottal_radii = [subglottal_radii]*len(subglottal_lengths)
            self.set_subglottal(lengths=subglottal_lengths,
                                radii=subglottal_radii)
        if supraglottal_lengths is not None:
            try:
                len(supraglottal_lengths)
            except TypeError:
                supraglottal_lengths = [supraglottal_lengths]
            try:
                len(supraglottal_radii)
            except TypeError:
                supraglottal_radii = [supraglottal_radii]*len(supraglottal_lengths)
            self.set_supraglottal(lengths=supraglottal_lengths,
                                  radii=supraglottal_radii)
    
    def set_supraglottal(self,lengths,radii, loss_multiplier=5):
        """
        set the dimensions of the supraglottal tract (mouth)
        
        lengths and radii in m
        loss_multiplier to increase propagation losses
        """
        ta = psyn.Duct()
        for rr, ll in zip(radii,lengths):
            ta.append_element(psyn.StraightDuct(length=ll,radius=rr,
                                                 loss_multiplier=loss_multiplier))
        ta.set_termination(self._supraglottal_term(radius=rr))
        self.supraglottal = ta

    def set_subglottal(self,lengths,radii, loss_multiplier=5):
        """
        set the dimensions of the subglottal tract (trachea)
        
        lengths and radii in m
        loss_multiplier to increase propagation losses
        """
        ta = psyn.Duct()
        for rr, ll in zip(radii,lengths):
            ta.append_element(psyn.StraightDuct(length=ll,radius=rr,
                                                 loss_multiplier=loss_multiplier))
        ta.set_termination(self._subglottal_term(radius=rr))
        self.subglottal = ta
    
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
        """
        Input impedance of vocal tract as seen from glottis
        looking outwards
        """
        return self.supraglottal.get_input_impedance_at_freq(f)

    def subglottal_impedance_from_glottis(self,f):
        """
        Input impedance of trachea as seen from glottis
        looking towards the lungs
        """
        return self.subglottal.get_input_impedance_at_freq(f)

    def transfer_function_flow_flow(self,f,no_subglottal=False):
        """
        Transfer function U_mouth/U_glottis in frequency domain

        use no_subglottal to remove tracheal zeros (equivalent
        to the transfer function of the vocal tract with glottis closed)
        """
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
        

