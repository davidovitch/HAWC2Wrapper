from numpy import zeros, array, pi
from vartrees import RotorVT, NacelleVT, GeneratorVT, TowerVT, ShaftVT, HubVT


class HAWC2AirfoilPolar(object):
    """A single airfoil polar"""
    def __init__(self):
        self.desc = ''
        self.rthick = 0.0
        self.aoa = zeros([1])
        self.cl = zeros([1])
        self.cd = zeros([1])
        self.cm = zeros([1])


class HAWC2AirfoilDataset(object):
    """A set of airfoil polars for a range of relative thicknesses"""
    def __init__(self):
        self.np = 0  # number of airfoil polars in set
        # Array of relative thicknesses linked to the airfoil polars
        self.rthick = zeros([1])
        self.polars = []  # List of polars


class HAWC2AirfoilData(object):
    """A list of airfoil datasets"""
    def __init__(self):
        self.nset = 0  # Number of airfoil datasets
        self.desc = ''  # String describing the airfoil data
        self.pc_sets = []  # List of airfoil datasets


class HAWC2BladeGeometry(object):

    def __init__(self):
        self.radius = 0.0
        self.s = zeros([1])  # Running length along blade axis
        self.c12axis = zeros([1])  # Pitch axis of blade
        self.chord = zeros([1])  # Blade chord
        self.rthick = zeros([1])  # Blade relative thickness
        self.twist = zeros([1])  # Blade twist (positive nose up!)
        self.aeset = zeros([1])  # Airfoil set


class HAWC2BeamStructure(object):

    def __init__(self):
        self.s = zeros([1])     # Running curve length of beam', units='m
        self.dm = zeros([1])    # Mass per unit length', units='kg/m
        self.x_cg = zeros([1])  # x-distance from blade axis to CM [m]
        self.y_cg = zeros([1])  # y-distance from blade axis to CM [m]
        self.ri_x = zeros([1])  # radius of gyration relative to EC [m]
        self.ri_y = zeros([1])  # radius of gyration relative to EC [m]
        self.x_sh = zeros([1])  # x-distance from blade axis to SC [m]
        self.y_sh = zeros([1])  # y-distance from blade axis to SC [m]
        self.E = zeros([1])     # modulus of elasticity [N/m**2]
        self.G = zeros([1])     # shear modulus of elasticity [N/m**2]
        # area moment of inertia w.r.t. principal bending xe axis [m4]
        self.I_x = zeros([1])
        # area moment of inertia w.r.t. principal bending ye axis [m4]
        self.I_y = zeros([1])
        # torsional stiffness constant w.r.t. ze axis at the SC[m4/rad]
        self.K = zeros([1])
        # shear factor for force in principal bending xe direction
        self.k_x = zeros([1])
        # shear factor for force in principal bending ye direction
        self.k_y = zeros([1])
        self.A = zeros([1])    # cross sectional area [m**2]
        self.pitch = zeros([1])  # structural pitch relative to main axis.[deg]
        self.x_e = zeros([1])   # x-distance from main axis to EC [m]
        self.y_e = zeros([1])   # y-distance from main axis to EC [m]
        self.K_11 = zeros([1])  # Elem. 1,1 of Constitutive Mat. [N*m**2]
        self.K_12 = zeros([1])  # Elem. 1,2 of Constitutive Mat. [N*m**2]
        self.K_13 = zeros([1])  # Elem. 1,3 of Constitutive Mat. [N*m**2]
        self.K_14 = zeros([1])  # Elem. 1,4 of Constitutive Mat. [N*m**2]
        self.K_15 = zeros([1])  # Elem. 1,5 of Constitutive Mat. [N*m**2]
        self.K_16 = zeros([1])  # Elem. 1,6 of Constitutive Mat. [N*m**2]
        self.K_22 = zeros([1])  # Elem. 2,2 of Constitutive Mat. [N*m**2]
        self.K_23 = zeros([1])  # Elem. 2,3 of Constitutive Mat. [N*m**2]
        self.K_24 = zeros([1])  # Elem. 2,4 of Constitutive Mat. [N*m**2]
        self.K_25 = zeros([1])  # Elem. 2,5 of Constitutive Mat. [N*m**2]
        self.K_26 = zeros([1])  # Elem. 2,6 of Constitutive Mat. [N*m**2]
        self.K_33 = zeros([1])  # Elem. 3,3 of Constitutive Mat. [N*m**2]
        self.K_34 = zeros([1])  # Elem. 3,4 of Constitutive Mat. [N*m**2]
        self.K_35 = zeros([1])  # Elem. 3,5 of Constitutive Mat. [N*m**2]
        self.K_36 = zeros([1])  # Elem. 3,6 of Constitutive Mat. [N*m**2]
        self.K_44 = zeros([1])  # Elem. 4,4 of Constitutive Mat. [N*m**2]
        self.K_45 = zeros([1])  # Elem. 4,5 of Constitutive Mat. [N*m**2]
        self.K_46 = zeros([1])  # Elem. 4,6 of Constitutive Mat. [N*m**2]
        self.K_55 = zeros([1])  # Elem. 5,5 of Constitutive Mat. [N*m**2]
        self.K_56 = zeros([1])  # Elem. 5,6 of Constitutive Mat. [N*m**2]
        self.K_66 = zeros([1])  # Elem. 6,6 of Constitutive Mat. [N*m**2]


class HAWC2OrientationBase(object):

    def __init__(self):
        self.body = ''  # mbdy name
        self.inipos = zeros(3)  # Initial position in global coordinates
        self.body_eulerang = []  # sequence of euler angle rotations, x->y->z


class HAWC2OrientationRelative(object):

    def __init__(self):
        self.body1 = []  # Main body name to which the body is attached
        self.body2 = []  # Main body name to which the body is attached
        self.body2_eulerang = []  # sequence of euler angle rotations, x->y->z
        # Initial rotation velocity of main body and all subsequent attached 
        # bodies (vx, vy, vz, |v|)
        self.mbdy2_ini_rotvec_d1 = zeros(4)


class HAWC2Constraint(object):

    def __init__(self):
        self.con_name = ''
        self.con_type = 'free'  # 'fixed', 'fixed_to_body', 'free', 'prescribed_angle'
        self.body1 = ''         # Main body name to which the body is attached
        self.DOF = zeros(6)  # Degrees of freedom


class HAWC2ConstraintFix0(object):

    def __init__(self):
        self.con_type = ''
        self.mbdy = ''  # Main body name
        self.disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintFix1(object):

    def __init__(self):
        self.con_type = ''
        self.mbdy1 = []  # Main_body name to which the next main_body is fixed
        self.mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
        self.disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintFix23(object):

    def __init__(self):
        self.con_type = ''
        self.mbdy = ''  # Main_body name to which the next main_body is fixed
        # Direction in global coo that is fixed in rotation 0: free, 1: fixed
        self.dof = zeros(3)


class HAWC2ConstraintFix4(object):

    def __init__(self):
        self.con_type = ''
        self.mbdy1 = []  # Main_body name to which the next main_body is fixed
        self.mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
        self.time = 2.  # Time for the pre-stress process. Default=2 sec


class HAWC2ConstraintBearing45(object):

    def __init__(self):
        self.name = ''
        self.con_type = ''
        self.mbdy1 = []  # Main_body name to which the next main_body is fixed
        self.mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
        # Vector to which the free rotation is possible. The direction of this
        # vector also defines the coo to which the output angle is defined
        #  1. Coo. system used for vector definition (0=global,1=mbdy1,2=mbdy2)
        #  2. x-axis
        #  3. y-axis
        #  4. z-axis
        self.bearing_vector = zeros(4)


class HAWC2ConstraintBearing12(HAWC2ConstraintBearing45):

    def __init__(self):
        super(HAWC2ConstraintBearing12, self).__init__()
        self.disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintBearing3(HAWC2ConstraintBearing45):

    def __init__(self):
        super(HAWC2ConstraintBearing3, self).__init__()
        self.omegas = 0.0  # Rotational speed


con_dict = {'fix0': HAWC2ConstraintFix0,
            'fix1': HAWC2ConstraintFix1,
            'fix2': HAWC2ConstraintFix23,
            'fix3': HAWC2ConstraintFix23,
            'fix4': HAWC2ConstraintFix4,
            'bearing1': HAWC2ConstraintBearing12,
            'bearing2': HAWC2ConstraintBearing12,
            'bearing3': HAWC2ConstraintBearing3,
            'bearing4': HAWC2ConstraintBearing45,
            'bearing5': HAWC2ConstraintBearing45}


class HAWC2MainBody(object):

    def __init__(self):
        self.body_name = 'body'
        self.body_type = 'timoschenko'
        self.st_filename = ''
        self.st_input_type = 1
        self.beam_structure = []
        self.body_set = [1, 1]  # Index of beam structure set to use from st file
        self.nbodies = 1
        self.node_distribution = 'c2_def'
        self.damping_type = ''
        self.damping_posdef = zeros(6)
        self.damping_aniso = zeros(6)
        self.copy_main_body = ''
        self.c12axis = zeros([1])  # C12 axis containing (x_c12, y_c12, z_c12, twist)
        self.concentrated_mass = []
        self.body_set = array([1, 1])
        self.orientations = []
        self.constraints = []

    def add_orientation(self, orientation):
        if orientation == 'base':
            o = HAWC2OrientationBase()
            self.orientations.append(o)
            return o
        elif orientation == 'relative':
            o = HAWC2OrientationRelative()
            self.orientations.append(o)
            return o

    def add_constraint(self, con_type, con_name='con', body1='empty',
                       DOF=zeros(6)):
        """
        add a constraint
        """

        con = HAWC2Constraint()
        con.con_type = con_type
        if con_type == 'fixed_to_body':
            con.body1 = body1
        if con_type in ('free', 'prescribed_angle'):
            if con_name is None:
                raise RuntimeError('Contraint name not specified')
            con.DOF = DOF
            con.con_name = con_name
            con.body1 = body1

        self.constraints.append(con)
        return con

    def add_constraint_new(self, con_type):
        """
        add a constraint
        """

        klass = con_dict[con_type]
        con = klass()
        self.constraints.append(con)
        con.con_type = con_type
        return con


class HAWC2MainBodyList(object):

    def add_main_body(self, body_name, b):

        b.body_name = body_name
        if not hasattr(self, body_name):
            setattr(self, body_name, b)
        else:
            print 'body: %s already added' % body_name

    def remove_main_body(self, body_name):

        try:
            setattr(self, body_name, None)
        except:
            raise RuntimeError('Body %s not in list of bodies' % body_name)

    def get_main_body(self, body_name, add_if_missing=False):

        if not hasattr(self, body_name):
            if add_if_missing:
                self.add_main_body(body_name, HAWC2MainBody())
            else:
                raise RuntimeError('Error: Main body %s not present' %
                                   body_name)

        return getattr(self, body_name)


class HAWC2Simulation(object):

    def __init__(self):
        self.time_stop = 300.  # Simulation stop time
        self.solvertype = 1    # Solver type. Default Newmark
        self.convergence_limits = [1.0e3, 1.0, 0.7]  # Sovler convergence limits
        self.on_no_convergence = ''
        self.max_iterations = 100   # Maximum iterations
        self.newmark_deltat = 0.02  # Newmark time step
        self.eig_out = False
        self.logfile = 'hawc2_case.log'


class HAWC2Aero(object):

    def __init__(self):
        self.nblades = 3  # Number of blades
        self.hub_vec_mbdy_name = 'shaft'
        self.hub_vec_coo = -3
        self.links = []
        self.induction_method = 1  # (0, 1) induction method, 0=none, 1=normal
        self.aerocalc_method = 1   # (0, 1) aero method, 0=none, 1=normal
        self.aerosections = 30     # Number of BEM aerodynamic sections
        self.tiploss_method = 1    # tiploss method, 0=none, 1=prandtl
        self.dynstall_method = 2   # dynamicstall method 0=none, 1=oeye, 2=mhh
        self.ae_sets = [1, 1, 1]
        self.ae_filename = ''
        self.pc_filename = ''


class HAWC2Mann(object):

    def __init__(self):
        self.create_turb = True
        self.L = 29.4
        self.alfaeps = 1.0
        self.gamma = 3.7
        self.seed = 0.0
        self.highfrq_compensation = 1.
        self.turb_base_name = 'mann_turb'
        self.turb_directory = 'turb'
        self.box_nu = 8192
        self.box_nv = 32
        self.box_nw = 32
        self.box_du = 0.8056640625
        self.box_dv = 5.6
        self.box_dw = 5.6
        self.std_scaling = array([1.0, 0.7, 0.5])


class HAWC2Wind(object):

    def __init__(self):
        self.density = 1.225  # Density
        self.wsp = 0.0   # Hub height wind speed
        self.tint = 0.0  # Turbulence intensity
        self.horizontal_input = 1  # 0=meteorological default, 1=horizontal
        # Turbulence box center point in global coordinates
        self.center_pos0 = zeros(3)
        # Orientation of the wind field, yaw, tilt, rotation.
        self.windfield_rotations = zeros(3)
        # Definition of the mean wind shear:
        # 0=None,1=constant,2=logarithmic,3=power law,4=linear
        self.shear_type = 1
        self.shear_factor = 0.  # Shear parameter - depends on the shear type
        self.turb_format = 0    # Turbulence format (0=none, 1=mann, 2=flex)
        # Tower shadow model 0=none, 1=potential, 2=jet model, 3=potential_2
        self.tower_shadow_method = 0
        self.scale_time_start = 0.0   # Starting time for turbulence scaling
        self.wind_ramp_t0 = 0.   # Start time for wind ramp
        self.wind_ramp_t1 = 0.0  # End time for wind ramp
        self.wind_ramp_factor0 = 0.  # Start factor for wind ramp
        self.wind_ramp_factor1 = 1.  # End factor for wind ramp
        self.wind_ramp_abs = []
        self.iec_gust = False       # Flag for specifying an IEC gust
        self.iec_gust_type = 'eog'  # 'eog','edc','ecd','ews   # IEC gust types
        self.G_A = 0.0
        self.G_phi0 = 0.0
        self.G_t0 = 0.0
        self.G_T = 0.0
        self.mann = HAWC2Mann()


class HAWC2TowerPotential2(object):

    def __init__(self):
        self.tower_mbdy_link = 'tower'
        self.nsec = 1
        self.sections = []


class HAWC2AeroDrag(object):

    def __init__(self):
        self.elements = []


class HAWC2AeroDragElement(object):

    def __init__(self):
        self.mbdy_name = ''
        self.dist = ''
        self.nsec = 1
        self.sections = []


class HAWC2OutputListVT(object):

    def __init__(self):
        self.sensor_list = []

    def set_outputs(self, entries):

        for i, c in enumerate(entries):
            if c.name in ['filename', 'time', 'data_format', 'buffer']:
                continue
            self.sensor_list.append(c.name + ' ' + ' '.join(
                [str(val) for val in _makelist(c.val)]))
            setattr(self, 'out_%i' % (i + 1), zeros([1]))


class HAWC2OutputVT(HAWC2OutputListVT):

    def __init__(self):
        super(HAWC2OutputVT, self).__init__()
        self.time_start = 0.
        self.time_stop = 0.0
        self.out_format = 'hawc_ascii'
        self.out_buffer = 1


class HAWC2Type2DLLinit(object):

    def set_constants(self, constants):

        for i, c in enumerate(constants):
            self.add('constant%i' % (i + 1), c.val[1])


def _makelist(val):

    if isinstance(val, list):
        return val
    else:
        return [val]


class HAWC2Type2DLLoutput(object):

    def set_outputs(self, entries):

        for i, c in enumerate(entries):
            self.add('out_%i' % (i + 1), zeros([1]))


class HAWC2Type2DLL(object):

    def __init__(self):
        self.name = ''  # Reference name of DLL
        self.filename = ''  # Filename incl. relative path of the DLL
        self.dll_subroutine_init = ''  # Name of initialization subroutine in DLL
        # Name of subroutine in DLL that is addressed at every time step
        self.dll_subroutine_update = ''
        self.arraysizes_init = []    # size of array in the initialization call
        self.arraysizes_update = []  # size of array in the update call
        self.deltat = 0.0            # Time between dll calls.
        self.dll_init = HAWC2Type2DLLinit()   # Slot for DLL specific
        self.output = HAWC2Type2DLLoutput()   # Outputs for DLL specific
        self.actions = HAWC2Type2DLLoutput()  # Actions for DLL specific

    def set_init(self, name):
        """
        sets the parameters slot with the VariableTree corresponding to the
        name string defined in the type2_dll interface
        """
        try:
            klass = type2_dll_dict[name]
            self.dll_init = klass()
        except:
            self._logger.warning('No init vartree available for %s, falling \
                                back on default HAWC2Type2DLLinit' % self.name)
        return self.dll_init

    def set_output(self, name):
        """
        sets the parameters slot with the VariableTree corresponding to the
        name string defined in the type2_dll interface
        """
        try:
            klass = type2_dll_out_dict[name]  # TODO:
            self.output = klass()
        except:
            self._logger.warning('No output vartree available for %s, falling \
                            back on default HAWC2Type2DLLoutput' % self.name)
        return self.output

    def set_actions(self, name):
        """
        sets the parameters slot with the VariableTree corresponding to the
        name string defined in the type2_dll interface
        """
        try:
            klass = type2_dll_action_dict[name]  # TODO:
            self.actions = klass()
        except:
            self._logger.warning('No actions vartree available for %s, falling\
                            back on default HAWC2Type2DLLoutput' % self.name)
        return self.actions


class HAWC2Type2DLLList(object):

    def add_dll(self, dll_name, b):

        b.dll_name = dll_name
        if not hasattr(self, dll_name):
            setattr(self, dll_name, b)
        else:
            print 'dll: %s already added' % dll_name


class DTUBasicControllerVT(HAWC2Type2DLLinit):
    """
    Variable tree for DTU Basic Controller inputs
    """
    def __init__(self):
        super(HAWC2Type2DLLinit, self).__init__()
        self.Vin = 4.   # [m/s]
        self.Vout = 25.  # [m/s]
        self.nV = 22
        self.ratedPower = 0.0  # [W]
        self.ratedAeroPower = 0.0  # [W]
        self.minRPM = 0.0  # [rpm]
        self.maxRPM = 0.0  # [rpm]
        self.gearRatio = 0.0
        self.designTSR = 7.5
        self.active = True
        self.FixedPitch = False
        self.maxTorque = 15.6e6   # Maximum allowable generator torque [N*m]
        self.minPitch = 100.      # minimum pitch angle [deg]
        self.maxPitch = 90.       # maximum pith angle [deg]
        self.maxPitchSpeed = 10.  # Maximum pitch velocity operation [deg/s]
        self.maxPitchAcc = 8.
        self.generatorFreq = 0.2     # Frequency of generator speed filter [Hz]
        self.generatorDamping = 0.7  # Damping ratio of speed filter
        self.ffFreq = 1.85    # Frequency of free-free DT torsion mode [Hz]
        self.Qg = 0.1001E+08  # Optimal Cp tracking K factor [kN*m/(rad/s)**2]
        self.pgTorque = 0.683E+08  # Proportional gain of torque contr.[Nm/(rad/s)]
        self.igTorque = 0.153E+08  # Integral gain of torque contr.[N*m/rad]
        self.dgTorque = 0.  # Differential gain of torque contr.[N*m/(rad/s**2)]
        self.pgPitch = 0.524E+00  # Proportional gain of torque contr.[N*m/(rad/s)]
        self.igPitch = 0.141E+00  # Integral gain of torque contr.[N*m/rad]
        self.dgPitch = 0.  # Differential gain of torque contr.[N*m/(rad/s**2)]
        self.prPowerGain = 0.4e-8   # Proportional power error gain
        self.intPowerGain = 0.4e-8  # Proportional power error gain
        # Generator control switch 1=constant power, 2=constant torque
        self.generatorSwitch = 1
        self.KK1 = 198.32888  # Coefficient of linear term in aero GS
        self.KK2 = 693.22213  # Coefficient of quadratic term in aero GS
        self.nlGainSpeed = 1.3  # Relative speed for double nonlinear gain
        self.softDelay = 4.     # Time delay for soft start of torque
        self.cutin_t0 = 0.1
        self.stop_t0 = 860.
        self.TorqCutOff = 5.
        self.PitchDelay1 = 1.
        self.PitchVel1 = 1.5
        self.PitchDelay2 = 1.
        self.PitchVel2 = 2.04
        self.generatorEfficiency = 0.94
        self.overspeed_limit = 1500.
        self.minServoPitch = 0         # maximum pith angle [deg]
        self.maxServoPitchSpeed = 30.  # Maximum pitch velocity [deg/s]
        self.maxServoPitchAcc = 8.
        self.poleFreqTorque = 0.05
        self.poleDampTorque = 0.7
        self.poleFreqPitch = 0.1
        self.poleDampPitch = 0.7
        self.gainScheduling = 2  # Gain scheduling [1: linear, 2: quadratic]
        self.prvs_turbine = 0    # [0: pitch regulated, 1: stall regulated]
        self.rotorspeed_gs = 0   # Gain scheduling [0:standard, 1:with damping]
        self.Kp2 = 0.0  # Additional GS param. kp_speed
        self.Ko1 = 1.0  # Additional GS param. invkk1_speed
        self.Ko2 = 0.0  # Additional GS param. invkk2_speed

    def set_constants(self, constants):

        for i, c in enumerate(constants):
            if   c.val[0] ==  1: self.ratedPower = c.val[1] * 1.e3
            elif c.val[0] ==  2: self.minRPM = c.val[1] * 60./(2.*pi)
            elif c.val[0] ==  3: self.maxRPM = c.val[1] * 60./(2.*pi)
            elif c.val[0] ==  4: self.maxTorque = c.val[1]
            elif c.val[0] ==  5: self.minPitch = c.val[1]
            elif c.val[0] ==  6: self.maxPitch = c.val[1]
            elif c.val[0] ==  7: self.maxPitchSpeed = c.val[1]
            elif c.val[0] ==  8: self.generatorFreq = c.val[1]
            elif c.val[0] ==  9: self.generatorDamping = c.val[1]
            elif c.val[0] == 10: self.ffFreq = c.val[1]
            elif c.val[0] == 11: self.Qg = c.val[1]
            elif c.val[0] == 12: self.pgTorque = c.val[1]
            elif c.val[0] == 13: self.igTorque = c.val[1]
            elif c.val[0] == 14: self.dgTorque = c.val[1]
            elif c.val[0] == 15: self.generatorSwitch = c.val[1]
            elif c.val[0] == 16: self.pgPitch  = c.val[1]
            elif c.val[0] == 17: self.igPitch  = c.val[1]
            elif c.val[0] == 18: self.dgPitch  = c.val[1]
            elif c.val[0] == 19: self.prPowerGain = c.val[1]
            elif c.val[0] == 20: self.intPowerGain = c.val[1]
            elif c.val[0] == 21: self.KK1 = c.val[1]
            elif c.val[0] == 22: self.KK2 = c.val[1]
            elif c.val[0] == 23: self.nlGainSpeed = c.val[1]
            elif c.val[0] == 24: self.cutin_t0 = c.val[1]
            elif c.val[0] == 25: self.softDelay = c.val[1]
            elif c.val[0] == 26: self.stop_t0 = c.val[1]
            elif c.val[0] == 27: self.TorqCutOff = c.val[1]
            elif c.val[0] == 29: self.PitchDelay1= c.val[1]
            elif c.val[0] == 30: self.PitchVel1  = c.val[1]
            elif c.val[0] == 31: self.PitchDelay2= c.val[1]
            elif c.val[0] == 32: self.PitchVel2  = c.val[1]
            elif c.val[0] == 39: self.overspeed_limit = c.val[1]

            # pick up the rest of the controller constants in generic variables
            else:
                self.add('constant%i' % (i + 1), c.val[1])


type2_dll_dict = {'risoe_controller': DTUBasicControllerVT}


class HAWC2SCommandsOpt(object):

    def __init__(self):
        self.include_torsiondeform = 1
        self.bladedeform = 'bladedeform'
        self.tipcorrect = 'tipcorrect'
        self.induction = 'induction'
        self.gradients = 'gradients'
        self.blade_only = False
        self.matrixwriteout = 'nomatrixwriteout'
        self.eigenvaluewriteout = 'noeigenvaluewriteout'
        self.frequencysorting = 'modalsorting'
        self.number_of_modes = 10
        self.maximum_damping = 0.5
        self.minimum_frequency = 0.5
        self.zero_pole_threshold = 0.1
        self.aero_deflect_ratio = 0.01
        self.vloc_out = False
        self.regions = zeros([1])
        self.remove_torque_limits = 0


class HAWC2SBody(object):

    def __init__(self):
        self.main_body = []
        self.log_decrements = zeros(6)


class SecondOrderActuator(object):

    def __init__(self):
        self.name = 'pitch1'
        self.frequency = 100.
        self.damping = 0.9


class HAWC2SVar(object):

    def __init__(self):
        self.ground_fixed = HAWC2SBody()
        self.rotating_axissym = HAWC2SBody()
        self.rotating_threebladed = HAWC2SBody()
        self.second_order_actuator = SecondOrderActuator()
        self.commands = []
        self.options = HAWC2SCommandsOpt()
        self.operational_data_filename = ''
        self.ch_list_in = HAWC2OutputListVT()
        self.ch_list_out = HAWC2OutputListVT()
        self.wsp_curve = zeros([1])  # Pitch curve from operational data file
        self.pitch_curve = zeros([1])  # Pitch curve from operational data file
        self.rpm_curve = zeros([1])  # RPM curve from operational data file
        self.wsp_cases = []
        self.cases = []  # List of input dictionaries with wsp, rpm and pitch


class HAWC2VarTrees(object):

    def __init__(self):
        self.sim = HAWC2Simulation()
        self.wind = HAWC2Wind()
        self.aero = HAWC2Aero()
        self.aerodrag = HAWC2AeroDrag()
        self.blade_ae = HAWC2BladeGeometry()
        self.blade_structure = []
        self.airfoildata = HAWC2AirfoilData()
        self.output = HAWC2OutputVT()
        self.rotor = RotorVT()
        """
        self.nacelle = NacelleVT()
        self.generator = GeneratorVT()
        self.tower = TowerVT()
        self.shaft = ShaftVT()
        self.hub = HubVT()
        """
        self.body_order = []
        self.main_bodies = HAWC2MainBodyList()
        self.dlls = HAWC2Type2DLLList()
        self.h2s = HAWC2SVar()
