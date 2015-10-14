from numpy import zeros, array, pi
from vartrees import RotorVT, NacelleVT, GeneratorVT, TowerVT, ShaftVT, HubVT


class HAWC2AirfoilPolar(object):
    """A single airfoil polar"""

    desc = ''
    rthick = 0.0
    aoa = zeros([1])
    cl = zeros([1])
    cd = zeros([1])
    cm = zeros([1])


class HAWC2AirfoilDataset(object):
    """A set of airfoil polars for a range of relative thicknesses"""

    np = 0  # number of airfoil polars in set
    # Array of relative thicknesses linked to the airfoil polars
    rthick = zeros([1])
    polars = []  # List of polars


class HAWC2AirfoilData(object):
    """A list of airfoil datasets"""

    nset = 0  # Number of airfoil datasets
    desc = ''  # String describing the airfoil data
    pc_sets = []  # List of airfoil datasets


class HAWC2BladeGeometry(object):

    radius = 0.0
    s = zeros([1])  # Running length along blade axis
    c12axis = zeros([1])  # Pitch axis of blade
    chord = zeros([1])  # Blade chord
    rthick = zeros([1])  # Blade relative thickness
    twist = zeros([1])  # Blade twist (positive nose up!)
    aeset = zeros([1])  # Airfoil set


class HAWC2BeamStructure(object):

    s = zeros([1])     # Running curve length of beam', units='m
    dm = zeros([1])    # Mass per unit length', units='kg/m
    x_cg = zeros([1])  # x-distance from blade axis to center of mass [m]
    y_cg = zeros([1])  # y-distance from blade axis to center of mass [m]
    ri_x = zeros([1])  # radius of gyration relative to elastic center. [m]
    ri_y = zeros([1])  # radius of gyration relative to elastic center [m]
    x_sh = zeros([1])  # x-distance from blade axis to shear center [m]
    y_sh = zeros([1])  # y-distance from blade axis to shear center [m]
    E = zeros([1])     # modulus of elasticity [N/m**2]
    G = zeros([1])     # shear modulus of elasticity [N/m**2]
    # area moment of inertia w.r.t. principal bending xe axis [m4]
    I_x = zeros([1])
    # area moment of inertia w.r.t. principal bending ye axis [m4]
    I_y = zeros([1])
    # torsional stiffness constant w.r.t. ze axis at the shear center [m4/rad]
    K = zeros([1])
    # shear factor for force in principal bending xe direction
    k_x = zeros([1])
    # shear factor for force in principal bending ye direction
    k_y = zeros([1])
    A = zeros([1])    # cross sectional area [m**2]
    pitch = zeros([1])  # structural pitch relative to main axis. [deg]
    x_e = zeros([1])   # x-distance from main axis to elastic center [m]
    y_e = zeros([1])   # y-distance from main axis to elastic center [m]
    K_11 = zeros([1])  # Elem. 1,1 of Sectional Constitutive Mat. [N*m**2]
    K_12 = zeros([1])  # Elem. 1,2 of Sectional Constitutive Mat. [N*m**2]
    K_13 = zeros([1])  # Elem. 1,3 of Sectional Constitutive Mat. [N*m**2]
    K_14 = zeros([1])  # Elem. 1,4 of Sectional Constitutive Mat. [N*m**2]
    K_15 = zeros([1])  # Elem. 1,5 of Sectional Constitutive Mat. [N*m**2]
    K_16 = zeros([1])  # Elem. 1,6 of Sectional Constitutive Mat. [N*m**2]
    K_22 = zeros([1])  # Elem. 2,2 of Sectional Constitutive Mat. [N*m**2]
    K_23 = zeros([1])  # Elem. 2,3 of Sectional Constitutive Mat. [N*m**2]
    K_24 = zeros([1])  # Elem. 2,4 of Sectional Constitutive Mat. [N*m**2]
    K_25 = zeros([1])  # Elem. 2,5 of Sectional Constitutive Mat. [N*m**2]
    K_26 = zeros([1])  # Elem. 2,6 of Sectional Constitutive Mat. [N*m**2]
    K_33 = zeros([1])  # Elem. 3,3 of Sectional Constitutive Mat. [N*m**2]
    K_34 = zeros([1])  # Elem. 3,4 of Sectional Constitutive Mat. [N*m**2]
    K_35 = zeros([1])  # Elem. 3,5 of Sectional Constitutive Mat. [N*m**2]
    K_36 = zeros([1])  # Elem. 3,6 of Sectional Constitutive Mat. [N*m**2]
    K_44 = zeros([1])  # Elem. 4,4 of Sectional Constitutive Mat. [N*m**2]
    K_45 = zeros([1])  # Elem. 4,5 of Sectional Constitutive Mat. [N*m**2]
    K_46 = zeros([1])  # Elem. 4,6 of Sectional Constitutive Mat. [N*m**2]
    K_55 = zeros([1])  # Elem. 5,5 of Sectional Constitutive Mat. [N*m**2]
    K_56 = zeros([1])  # Elem. 5,6 of Sectional Constitutive Mat. [N*m**2]
    K_66 = zeros([1])  # Elem. 6,6 of Sectional Constitutive Mat. [N*m**2]


class HAWC2OrientationBase(object):

    body = ''  # mbdy name
    inipos = zeros(3)  # Initial position in global coordinates
    body_eulerang = []  # sequence of euler angle rotations, x->y->z


class HAWC2OrientationRelative(object):

    body1 = []  # Main body name to which the body is attached
    body2 = []  # Main body name to which the body is attached
    body2_eulerang = []  # sequence of euler angle rotations, x->y->z
    # Initial rotation velocity of main body and all subsequent attached bodies
    # (vx, vy, vz, |v|)
    mbdy2_ini_rotvec_d1 = zeros(4)


class HAWC2Constraint(object):

    con_name = ''
    con_type = 'free'  # 'fixed', 'fixed_to_body', 'free', 'prescribed_angle'
    body1 = ''         # Main body name to which the body is attached
    DOF = zeros(6)  # Degrees of freedom


class HAWC2ConstraintFix0(object):

    con_type = ''
    mbdy = ''  # Main body name
    disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintFix1(object):

    con_type = ''
    mbdy1 = []  # Main_body name to which the next main_body is fixed
    mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
    disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintFix23(object):

    con_type = ''
    mbdy = ''  # Main_body name to which the next main_body is fixed
    # Direction in global coo that is fixed in rotation 0: free, 1: fixed
    dof = zeros(3)


class HAWC2ConstraintFix4(object):

    con_type = ''
    mbdy1 = []  # Main_body name to which the next main_body is fixed
    mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
    time = 2.  # Time for the pre-stress process. Default=2 sec


class HAWC2ConstraintBearing45(object):

    con_type = ''
    mbdy1 = []  # Main_body name to which the next main_body is fixed
    mbdy2 = []  # Main_body name of the main_body that is fixed to main_body1
    # Vector to which the free rotation is possible. The direction of this
    # vector also defines the coo to which the output angle is defined
    #  1. Coo. system used for vector definition (0=global,1=mbdy1,2=mbdy2)
    #  2. x-axis
    #  3. y-axis
    #  4. z-axis
    bearing_vector = zeros(4)


class HAWC2ConstraintBearing12(HAWC2ConstraintBearing45):

    disable_at = 0.0  # Time at which constraint can be disabled


class HAWC2ConstraintBearing3(HAWC2ConstraintBearing45):

    omegas = 0.0  # Rotational speed


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

    body_name = 'body'
    body_type = 'timoschenko'
    st_filename = ''
    st_input_type = 1
    beam_structure = []
    body_set = [1, 1]  # Index of beam structure set to use from st file
    nbodies = 1
    node_distribution = 'c2_def'
    damping_type = ''
    damping_posdef = zeros(6)
    damping_aniso = zeros(6)
    copy_main_body = ''
    c12axis = zeros([1])  # C12 axis containing (x_c12, y_c12, z_c12, twist)
    concentrated_mass = []
    body_set = array([1, 1])
    orientations = []
    constraints = []

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

    def add(self, body_name, b):
        setattr(self, body_name, b)

    def add_main_body(self, body_name, b):

        b.body_name = body_name
        if not hasattr(self, body_name):
            self.add(body_name, b)
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

    time_stop = 300.  # Simulation stop time
    solvertype = 1    # Solver type. Default Newmark
    convergence_limits = [1.0e3, 1.0, 0.7]  # Sovler convergence limits
    on_no_convergence = ''
    max_iterations = 100   # Maximum iterations
    newmark_deltat = 0.02  # Newmark time step
    eig_out = False
    logfile = 'hawc2_case.log'


class HAWC2Aero(object):

    nblades = 3  # Number of blades
    hub_vec_mbdy_name = 'shaft'
    hub_vec_coo = -3
    links = []
    induction_method = 1  # (0, 1) BEM induction method, 0=none, 1=normal
    aerocalc_method = 1   # (0, 1)  BEM aero method, 0=none, 1=normal
    aerosections = 30     # Number of BEM aerodynamic sections
    tiploss_method = 1    # BEM tiploss method, 0=none, 1=prandtl
    dynstall_method = 2   # BEM dynamicstall method, 0=none, 1=stig oeye, 2=mhh
    ae_sets = [1, 1, 1]
    ae_filename = ''
    pc_filename = ''


class HAWC2Mann(object):

    create_turb = True
    L = 29.4
    alfaeps = 1.0
    gamma = 3.7
    seed = 0.0
    highfrq_compensation = 1.
    turb_base_name = 'mann_turb'
    turb_directory = 'turb'
    box_nu = 8192
    box_nv = 32
    box_nw = 32
    box_du = 0.8056640625
    box_dv = 5.6
    box_dw = 5.6
    std_scaling = array([1.0, 0.7, 0.5])


class HAWC2Wind(object):

    density = 1.225  # Density
    wsp = 0.0   # Hub height wind speed
    tint = 0.0  # Turbulence intensity
    horizontal_input = 1  # 0=meteorological default, 1=horizontal
    # Turbulence box center point in global coordinates
    center_pos0 = zeros(3)
    # Orientation of the wind field, yaw, tilt, rotation.
    windfield_rotations = zeros(3)
    # Definition of the mean wind shear:
    # 0=None,1=constant,2=logarithmic,3=power law,4=linear
    shear_type = 1
    shear_factor = 0.  # Shear parameter - depends on the shear type
    turb_format = 0    # Turbulence format (0=none, 1=mann, 2=flex)
    # Tower shadow model 0=none, 1=potential flow, 2=jet model, 3=potential_2
    tower_shadow_method = 0
    scale_time_start = 0.0   # Starting time for turbulence scaling
    wind_ramp_t0 = 0.   # Start time for wind ramp
    wind_ramp_t1 = 0.0  # End time for wind ramp
    wind_ramp_factor0 = 0.  # Start factor for wind ramp
    wind_ramp_factor1 = 1.  # End factor for wind ramp
    wind_ramp_abs = []
    iec_gust = False       # Flag for specifying an IEC gust
    iec_gust_type = 'eog'  # 'eog','edc','ecd','ews   # IEC gust types
    G_A = 0.0
    G_phi0 = 0.0
    G_t0 = 0.0
    G_T = 0.0
    mann = HAWC2Mann()


class HAWC2TowerPotential2(object):

    tower_mbdy_link = 'tower'
    nsec = 1
    sections = []


class HAWC2AeroDrag(object):

    elements = []


class HAWC2AeroDragElement(object):

    mbdy_name = ''
    dist = ''
    nsec = 1
    sections = []


class HAWC2OutputListVT(object):

    sensor_list = []

    def set_outputs(self, entries):

        for i, c in enumerate(entries):
            if c.name in ['filename', 'time', 'data_format', 'buffer']:
                continue
            self.sensor_list.append(c.name + ' ' + ' '.join(
                [str(val) for val in _makelist(c.val)]))
            setattr(self, 'out_%i' % (i + 1), zeros([1]))


class HAWC2OutputVT(HAWC2OutputListVT):

    time_start = 0.
    time_stop = 0.0
    out_format = 'hawc_ascii'
    out_buffer = 1


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

    name = ''  # Reference name of DLL (to be used with DLL output commands)
    filename = ''  # Filename incl. relative path of the DLL
    dll_subroutine_init = ''  # Name of initialization subroutine in DLL
    # Name of subroutine in DLL that is addressed at every time step
    dll_subroutine_update = ''
    arraysizes_init = []    # size of array in the initialization call
    arraysizes_update = []  # size of array in the update call
    deltat = 0.0            # Time between dll calls.
    dll_init = HAWC2Type2DLLinit()   # Slot for DLL specific variable tree
    output = HAWC2Type2DLLoutput()   # Outputs for DLL specific variable tree
    actions = HAWC2Type2DLLoutput()  # Actions for DLL specific variable tree

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

    Vin = 4.   # [m/s]
    Vout = 25.  # [m/s]
    nV = 22

    ratedPower = 0.0  # [W]
    ratedAeroPower = 0.0  # [W]

    minRPM = 0.0  # [rpm]
    maxRPM = 0.0  # [rpm]
    gearRatio = 0.0
    designTSR = 7.5
    active = True
    FixedPitch = False

    maxTorque = 15.6e6   # Maximum allowable generator torque [N*m]
    minPitch = 100.      # minimum pitch angle [deg]
    maxPitch = 90.       # maximum pith angle [deg]
    maxPitchSpeed = 10.  # Maximum pitch velocity operation [deg/s]
    maxPitchAcc = 8.
    generatorFreq = 0.2     # Frequency of generator speed filter [Hz]
    generatorDamping = 0.7  # Damping ratio of speed filter
    ffFreq = 1.85    # Frequency of free-free DT torsion mode [Hz]
    Qg = 0.1001E+08  # Optimal Cp tracking K factor [kN*m/(rad/s)**2]
    pgTorque = 0.683E+08  # Proportional gain of torque controller [Nm/(rad/s)]
    igTorque = 0.153E+08  # Integral gain of torque controller [N*m/rad]
    dgTorque = 0.  # Differential gain of torque controller [N*m/(rad/s**2)]
    pgPitch = 0.524E+00  # Proportional gain of torque controller [N*m/(rad/s)]
    igPitch = 0.141E+00  # Integral gain of torque controller [N*m/rad]
    dgPitch = 0.  # Differential gain of torque controller [N*m/(rad/s**2)]
    prPowerGain = 0.4e-8   # Proportional power error gain
    intPowerGain = 0.4e-8  # Proportional power error gain
    # Generator control switch 1=constant power, 2=constant torque
    generatorSwitch = 1
    KK1 = 198.32888  # Coefficient of linear term in aero gain scheduling
    KK2 = 693.22213  # Coefficient of quadratic term in aero gain scheduling
    nlGainSpeed = 1.3  # Relative speed for double nonlinear gain
    softDelay = 4.     # Time delay for soft start of torque
    cutin_t0 = 0.1
    stop_t0 = 860.
    TorqCutOff = 5.
    PitchDelay1 = 1.
    PitchVel1 = 1.5
    PitchDelay2 = 1.
    PitchVel2 = 2.04
    generatorEfficiency = 0.94
    overspeed_limit = 1500.
    minServoPitch = 0         # maximum pith angle [deg]
    maxServoPitchSpeed = 30.  # Maximum pitch velocity operation [deg/s]
    maxServoPitchAcc = 8.

    poleFreqTorque = 0.05
    poleDampTorque = 0.7
    poleFreqPitch = 0.1
    poleDampPitch = 0.7

    gainScheduling = 2  # Gain scheduling [1: linear, 2: quadratic]
    prvs_turbine = 0    # [0: pitch regulated, 1: stall regulated]
    rotorspeed_gs = 0   # Gain scheduling [0:standard, 1:with damping]
    Kp2 = 0.0  # Additional gain-scheduling param. kp_speed
    Ko1 = 1.0  # Additional gain-scheduling param. invkk1_speed
    Ko2 = 0.0  # Additional gain-scheduling param. invkk2_speed

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

    include_torsiondeform = 1
    bladedeform = 'bladedeform'
    tipcorrect = 'tipcorrect'
    induction = 'induction'
    gradients = 'gradients'
    blade_only = False
    matrixwriteout = 'nomatrixwriteout'
    eigenvaluewriteout = 'noeigenvaluewriteout'
    frequencysorting = 'modalsorting'
    number_of_modes = 10
    maximum_damping = 0.5
    minimum_frequency = 0.5
    zero_pole_threshold = 0.1
    aero_deflect_ratio = 0.01
    vloc_out = False
    regions = zeros([1])
    remove_torque_limits = 0


class HAWC2SBody(object):

    main_body = []
    log_decrements = zeros(6)


class SecondOrderActuator(object):

    name = 'pitch1'
    frequency = 100.
    damping = 0.9


class HAWC2SVar(object):

    ground_fixed = HAWC2SBody()
    rotating_axissym = HAWC2SBody()
    rotating_threebladed = HAWC2SBody()
    second_order_actuator = SecondOrderActuator()
    commands = []
    options = HAWC2SCommandsOpt()
    operational_data_filename = ''
    ch_list_in = HAWC2OutputListVT()
    ch_list_out = HAWC2OutputListVT()

    wsp_curve = zeros([1])  # Pitch curve from operational data file
    pitch_curve = zeros([1])  # Pitch curve from operational data file
    rpm_curve = zeros([1])  # RPM curve from operational data file
    wsp_cases = []
    cases = []  # List of input dictionaries with wsp, rpm and pitch


class HAWC2VarTrees(object):

    sim = HAWC2Simulation()
    wind = HAWC2Wind()
    aero = HAWC2Aero()
    aerodrag = HAWC2AeroDrag()
    blade_ae = HAWC2BladeGeometry()
    blade_structure = []
    airfoildata = HAWC2AirfoilData()
    output = HAWC2OutputVT()
    rotor = RotorVT()
    nacelle = NacelleVT()
    generator = GeneratorVT()
    tower = TowerVT()
    shaft = ShaftVT()
    hub = HubVT()

    body_order = []
    main_bodies = HAWC2MainBodyList()
    dlls = HAWC2Type2DLLList()

    h2s = HAWC2SVar()
