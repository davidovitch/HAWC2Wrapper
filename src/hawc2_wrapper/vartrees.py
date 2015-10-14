from numpy import zeros


class RotorVT(object):

    hub_height = 0.0  # Hub height
    nblades = 3       # Number of blades
    tilt_angle = 0.0  # Tilt angle
    cone_angle = 0.0  # Cone angle
    diameter = 0.0    # Rotor diameter


class BladeVT(object):

    length = 0.0   # blade length
    mass = 0.0     # blade mass
    I_x = 0.0      # first area moment of inertia
    I_y = 0.0      # Second area moment of inertia
    root_chord = 0.0  # Blade root chord
    max_chord = 0.0   # Blade maximum chord
    tip_chord = 0.0   # Blade tip chord
    airfoils = []     # List of airfoil names used on blade


class HubVT(object):

    diameter = 0.0  # blade length
    mass = 0.0      # blade mass
    I_x = 0.0       # first area moment of inertia
    I_y = 0.0       # Second area moment of inertia
    CM = zeros(3)


class NacelleVT(object):

    mass = 0.0  # blade mass
    I_x = 0.0   # first area moment of inertia
    I_y = 0.0   # Second area moment of inertia
    CM = zeros(3)
    diameter = 0.0


class ShaftVT(object):

    mass = 0.0  # blade mass
    I_x = 0.0   # first area moment of inertia
    I_y = 0.0   # Second area moment of inertia
    CM = zeros(3)
    length = 0.0


class GeneratorVT(object):

    mass = 0.0        # blade mass
    I_x = 0.0         # first area moment of inertia
    I_y = 0.0         # Second area moment of inertia
    power = 0.0       # Generator power
    max_torque = 0.0  # Maximum allowable generator torque
    efficiency = 0.0  # Generator efficiency


class TransmissionVT(object):

    gear_ratio = 0.0  # Transmission gear ratio


class TowerVT():

    height = 0.0           # Tower height
    bottom_diameter = 0.0  # Tower bottom diameter
    top_diameter = 0.0     # Tower bottom diameter
    mass = 0.0             # Tower mass


class BeamGeometryVT(object):

    s = zeros([1])             # Blade main axis accumulated curve length (n)
    main_axis = zeros([1, 3])  # Blade main axis (n,3)
    rot_x = zeros([1])         # x-rotation angle (n)
    rot_y = zeros([1])         # y-rotation angle (n)
    rot_z = zeros([1])         # z-rotation angle (n)


class BladeGeometryVT(BeamGeometryVT):

    chord = zeros([1])   # Blade chord (n)
    rthick = zeros([1])  # Blade relative thickness (n)
    athick = zeros([1])  # Blade absolute thickness (n)
    p_le = zeros([1])    # normalized distance along chord line from
                            #  leading edge to main axis (n)


class DistributedLoadsVT(object):

    s = zeros([1])   # locations for distributed loads
    Fn = zeros([1])  # force per unit length in normal direction to the blade
    Ft = zeros([1])  # force per unit length in tangential direction to the blade


class DistributedLoadsExtVT(DistributedLoadsVT):

    cn = zeros([1])  # Normal force coefficient along the blade
    ct = zeros([1])  # Tangential force coefficient along the blade
    cl = zeros([1])  # Lift force coefficient along the blade
    cd = zeros([1])  # Drag force coefficient along the blade
    cm = zeros([1])  # Moment force coefficient along the blade
    aoa = zeros([1])  # [deg] Angle of attack along the blade
    lfa = zeros([1])  # [deg] Local flow angle along the blade
    v_a = zeros([1])  # [m/s] axial velocity along the blade
    v_t = zeros([1])  # [m/s] tangential velocity along the blade
    v_r = zeros([1])  # [m/s] radial velocity along the blade
    lcp = zeros([1])  # Local power coefficient along the blade
    lct = zeros([1])  # Local power coefficient along the blade


class RotorLoadsVT(object):

    T = 0.0  # [N] thrust
    Q = 0.0  # [N*m] torque
    P = 0.0  # [W] power

    CT = 0.0  # [N] thrust coefficient
    CQ = 0.0  # [N*m] torque coefficient
    CP = 0.0  # [W] power coefficient


class RotorLoadsArrayVT(object):

    wsp = zeros([1])  # [m/s] Wind speeds
    rpm = zeros([1])  # [rpm] Rotor speed
    pitch = zeros([1])  # [deg] Pitch angle

    T = zeros([1])  # [N] thrust
    Q = zeros([1])  # [N*m] torque
    P = zeros([1])  # [W] power

    CT = zeros([1])  # thrust coefficient
    CQ = zeros([1])  # torque coefficient
    CP = zeros([1])  # power coefficient


class DistributedLoadsArrayVT(object):

    loads_array = []


class BeamDisplacementsVT():

    main_axis = zeros([1])
    rot_x = zeros([1])
    rot_y = zeros([1])
    rot_z = zeros([1])


class BeamDisplacementsArrayVT(object):

    disps_array = []  # Array of blade displacements and rotations
    tip_pos = zeros([1])
    tip_rot = zeros([1])


class HubLoadsVT(object):

    Fx = 0.0  # [N] x-force in wind-aligned coordinate system
    Fy = 0.0  # [N] y-force in wind-aligned coordinate system
    Fz = 0.0  # [N] z-force in wind-aligned coordinate system
    Mx = 0.0  # [N*m] x-moment in wind-aligned coordinate system
    My = 0.0  # [N*m] y-moment in wind-aligned coordinate system
    Mz = 0.0  # [N*m] z-moment in wind-aligned coordinate system


class HubLoadsArrayVT(object):

    Fx = zeros([1])  # [N] x-force in wind-aligned coordinate system
    Fy = zeros([1])  # [N] y-force in wind-aligned coordinate system
    Fz = zeros([1])  # [N] z-force in wind-aligned coordinate system
    Mx = zeros([1])  # [N*m] x-moment in wind-aligned coordinate system
    My = zeros([1])  # [N*m] y-moment in wind-aligned coordinate system
    Mz = zeros([1])  # [N*m] z-moment in wind-aligned coordinate system


class RotorOperationalDataVT(object):

    wsp = zeros([1])    # [m/s] Wind speed
    pitch = zeros([1])  # [deg] pitch angle
    rpm = zeros([1])    # rotational speed
