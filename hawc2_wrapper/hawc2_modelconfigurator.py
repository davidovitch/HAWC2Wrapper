


class HAWC2ModelConfigurator(object):


    def configure_wt(self):

        self.configure_tower_body()
        self.configure_towertop_body()
        self.configure_shaft_body()
        self.configure_hub_bodies()
        self.configure_blade_bodies()

        self.add_tower_aerodrag()
        self.add_nacelle_aerodrag()

    def add_tower_aerodrag(self, cd=0.6):
        """convenience function to add tower drag"""
        geom = np.zeros((2, 2))
        geom[0, :] = 0, self.vartrees.tower.bottom_diameter
        geom[1, :] = self.vartrees.tower.height, self.vartrees.tower.top_diameter
        self.add_aerodrag_element('tower', geom, cd)

    def add_nacelle_aerodrag(self, cd=0.8, width=10.):
        """convenience function to add nacelle drag"""

        geom = np.zeros((2, 2))

        shaft = self.vartrees.main_bodies.get_main_body('shaft')

        geom[0, :] = 0, shaft.c12axis[0, 2]
        geom[1, :] = shaft.c12axis[-1, 2], width
        self.add_aerodrag_element('shaft', geom, cd)

    def configure_tower_body(self):
        """
        convenience method for adding tower body with orientation
        and constraints
        """
        b = self.vartrees.main_bodies.add_main_body('tower')
        b.c12axis = np.zeros((10, 4))
        b.c12axis[:, 2] = np.linspace(0, -self.tower.height, 10)
        b.add_orientation('base')
        b.orientations[0].eulerang.append(np.array([0, 0, 0]))
        b.add_constraint('fixed')
        print 'Not sure it makes sense configure_tower_body'  # FIXME:
        return b

    def configure_towertop_body(self):
        """
        convenience method for adding towertop body with orientation
        and constraints
        """

        b = self.vartrees.main_bodies.add_main_body('towertop')
        b.c12axis = np.zeros((2, 4))
        b.c12axis[-1, 2] = -self.vartrees.nacelle.diameter / 2.
        b.add_orientation('relative')
        b.orientations[0].mbdy1_name = 'tower'
        b.orientations[0].eulerang.append(np.array([0, 0, 0]))
        b.add_constraint('fixed_to_body', body1='tower')
        print 'Not sure it makes sense configure_towertop_body'  # FIXME:

    def configure_shaft_body(self):
        """
        convenience method for adding shaft body with orientation
        and constraints
        """

        b = self.vartrees.main_bodies.add_main_body('shaft')
        b.c12axis = np.zeros((5, 4))
        b.c12axis[:, 2] = np.linspace(0, self.vartrees.shaft.length, 5)
        b.add_orientation('relative')
        b.orientations[0].mbdy1_name = 'towertop'
        b.orientations[0].eulerang.append(np.array([90, 0, 0]))
        b.orientations[0].eulerang.append(np.array([self.rotor.tilt_angle,
                                                    0, 0]))
        b.orientations[0].mbdy2_ini_rotvec_d1 = 0.314
        b.orientations[0].rotation_dof = [0, 0, -1]
        b.add_constraint('free', body1='towertop', con_name='shaft_rot',
                         DOF=np.array([0, 0, 0, 0, 0, -1]))

    def configure_hub_bodies(self):
        """
        convenience method for adding hub bodies with orientation
        and constraints
        """

        b = self.vartrees.main_bodies.add_main_body('hub1')
        b.c12axis = np.zeros((2, 4))
        b.c12axis[1, 2] = self.vartrees.hub.diameter/2.
        b.nbodies = 1
        b.add_orientation('relative')
        b.orientations[0].mbdy1_name = 'shaft'
        b.orientations[0].eulerang.append(np.array([-90., 0., 0.]))
        b.orientations[0].eulerang.append(np.array([0., 180., 0.]))
        b.orientations[0].eulerang.append(
            np.array([self.vartrees.rotor.cone_angle, 0., 0.]))
        b.add_constraint('fixed_to_body', body1='shaft')

        for i in range(1, self.vartrees.rotor.nblades):
            b = self.vartrees.main_bodies.add_main_body('hub' + str(i + 1))
            b.copy_main_body = 'hub1'
            b.add_orientation('relative')
            b.orientations[0].mbdy1_name = 'shaft'
            b.orientations[0].eulerang.append(np.array([-90., 0., 0.]))
            b.orientations[0].eulerang.append(
                np.array([0., 60. - (i - 1) * 120., 0.]))
            b.orientations[0].eulerang.append(
                np.array([self.vartrees.rotor.cone_angle, 0., 0.]))
            b.add_constraint('fixed_to_body', body1='shaft')

    def configure_blade_bodies(self):
        """
        convenience method for adding blade bodies with orientation
        and constraints
        """

        b = self.vartrees.main_bodies.add_main_body('blade1')
        b.c12axis[:, :3] = self.vartrees.blade_ae.c12axis
        b.c12axis[:,  3] = self.vartrees.blade_ae.twist
        b.nbodies = 10
        b.add_orientation('relative')
        b.orientations[0].mbdy1_name = 'hub1'
        b.orientations[0].eulerang.append(np.array([0, 0, 0]))
        b.add_constraint('prescribed_angle', body1='hub1', con_name='pitch1',
                         DOF=np.array([0, 0, 0, 0, 0, -1]))

        for i in range(1, self.rotor.nblades):
            b = self.vartrees.main_bodies.add_main_body('blade' + str(i + 1))
            b.copy_main_body = 'blade1'
            b.add_orientation('relative')
            b.orientations[0].mbdy1_name = 'hub' + str(i + 1)
            b.orientations[0].eulerang.append(np.array([0, 0, 0]))
            b.add_constraint('prescribed_angle', body1='hub' + str(i + 1),
                             con_name='pitch' + str(i + 1),
                             DOF=np.array([0, 0, 0, 0, 0, -1]))

    def calculate_c12axis(self):
        """
        compute the 1/2 chord axis based on the blade axis and chordwise
        rotation point
        nb: this examples only works for straight blades! # FIXME:
        """

        # The HAWC2 blade axis is defined using the 1/2 chord points
        b = self.vartrees.blade_geom
        c12axis = np.zeros((b.main_axis.shape[0], 4))
        for i in range(b.main_axis.shape[0]):
            xc12 = (0.5 - b.p_le[i]) * b.chord[i] *\
                    np.cos(b.rot_z[i] * np.pi / 180.)
            yc12 = -(0.5 - b.p_le[i]) * b.chord[i] * np.sin(b.rot_z[i] * np.pi / 180.)
            c12axis[i, 0] = -(b.main_axis[i, 0] + xc12)
            c12axis[i, 1] = b.main_axis[i, 1] + yc12
            c12axis[i, 2] = b.main_axis[i, 2] - b.main_axis[0, 2]
        c12axis[:, 3] = b.rot_z
        return c12axis
        
    def update_c12axis(self):

        self.vartrees.main_bodies.blade1.c12axis = \
            self.vartrees.blade_ae.c12axis.copy()
        self.vartrees.main_bodies.blade1.beam_structure = \
            self.vartrees.blade_structure
            
            
    def _write_operational_data_file(self):

        h2s = self.vartrees.h2s
        ctrl = self.vartrees.dlls.risoe_controller.dll_init

        # HAWCStab2 will compute the operational point for us
        wsp = []
        pitch = []
        rpm = []
        if 'compute_optimal_pitch_angle' in self.vartrees.h2s.commands \
           and len(h2s.wsp_cases) > 0:

            ctrl.Vin = h2s.wsp_cases[0]
            ctrl.Vout = h2s.wsp_cases[-1]
            ctrl.nV = len(h2s.wsp_cases)

        # operational point is interpolated from the .opt file
        elif h2s.wsp_curve.shape[0] > 0:

            for w in h2s.wsp_cases:
                if self.set_tsr_flag:
                    minRPM = ctrl.minRPM / ctrl.gearRatio
                    maxRPM = ctrl.maxRPM / ctrl.gearRatio
                    omega = ctrl.designTSR * w / self.vartrees.blade_ae.radius
                    r = max(minRPM, min(maxRPM, omega * 60 / (2. * np.pi)))
                else:
                    r = np.interp(w, h2s.wsp_curve, h2s.rpm_curve)
                p = np.interp(w, h2s.wsp_curve, h2s.pitch_curve)
                wsp.append(w)
                pitch.append(p)
                rpm.append(r)

        for case in h2s.cases:
            try:
                wsp.append(case['wsp'])
                pitch.append(case['pitch'])
                rpm.append(case['rpm'])

            except:
                raise RuntimeError('wrong inputs in case')

        if len(wsp) > 0:
            data = np.array([wsp, pitch, rpm]).T
            ctrl.Vin = wsp[0]
            ctrl.Vout = wsp[-1]
            ctrl.nV = len(wsp)
        else:
            data = np.array([h2s.wsp_curve, h2s.pitch_curve, h2s.rpm_curve]).T

        fid = open(self.case_id + '.opt', 'w')
        fid.write(('%i Wind speed [m/s]          Pitch [deg]     ' +
                  'Rot. speed [rpm]\n') % data.shape[0])
        np.savetxt(fid, data)
        fid.close()
        
class H2SResInterp(object):

    def __init__(self):

        self.blade_loads = DistributedLoadsArrayVT()
        self.blade_disps = BeamDisplacementsArrayVT()
        self.rotor_loads = RotorLoadsArrayVT()
        self.hub_loads = PointLoadArray()
        self.oper = RotorOperationalDataArray()

        self.blade_loads_i = DistributedLoadsArrayVT()
        self.blade_disps_i = BeamDisplacementsArrayVT()
        self.rotor_loads_i = RotorLoadsArrayVT()
        self.hub_loads_i = PointLoadArray()
        self.oper_i = RotorOperationalDataArray()

        self.N = 50

        self.cutout_ws = 25.

    def execute(self):

        # Find rated wind speed
        P = self.rotor_loads.P
        ws = self.rotor_loads.wsp

        # Let's skip the interpolation if we only have one wind speed
        if len(ws) > 1:
            P_diff = (P[1:]-P[:-1])
            P_rated = P[P_diff.tolist().index(min(P_diff))]

            for ip, p in enumerate(P):
                if (P_rated - p)/P_rated < 0.01:
                    break
            ws_i = np.linspace(np.min(ws), ws[ip], 1e3)
            pp = np.polyfit(ws[:ip], P[:ip], 2)
            P_il = np.polyval(pp, ws_i)
            P_ir = P_rated*np.ones(len(ws_i))
            diff = np.abs(P_ir-P_il)
            ws_r = ws_i[diff.tolist().index(np.min(diff))]

            # Blade loads are not interpolated!
            self.blade_loads_i = self.blade_loads

            max_ws = max(self.rotor_loads.wsp)

            for iws, ws in enumerate(self.rotor_loads.wsp):
                if ws > self.cutout_ws:
                    max_ws = self.rotor_loads.wsp[iws-1]
                    break

            i_max_ws = self.rotor_loads.wsp.tolist().index(max_ws)+1

            self._logger.info('Number of wind speed in operative range: %i\
                               out of: %i' % (i_max_ws, len(self.rotor_loads.wsp)))
            n = self.N+(len(self.rotor_loads.wsp)-i_max_ws)

            self.rotor_loads_i.wsp = np.zeros([n])

            ws_i = np.linspace(min(self.rotor_loads.wsp), max_ws, self.N)

            self.rotor_loads_i.wsp[:self.N] = ws_i
            for i, v in enumerate(self.rotor_loads.wsp[i_max_ws:]):
                self.rotor_loads_i.wsp[self.N+i] = v

            # rotor_loads
            for att in self.rotor_loads.list_vars():
                if att[0] is '_':
                    continue
                if att is 'wsp':
                    continue
                val = getattr(self.rotor_loads, att)
                if not val.tolist():
                    continue
                val_i = np.zeros([n])

                val_i[:self.N] = pchip(self.rotor_loads.wsp[:i_max_ws],
                                       val[:i_max_ws])(ws_i)
                if att is 'T':
                    val_i[:self.N] = sharp_curves_interpolation(
                        self.rotor_loads.wsp[:i_max_ws],
                        val[:i_max_ws], ws_i, ws_r)
                for i, v in enumerate(val[i_max_ws:]):
                    val_i[self.N+i] = v
                setattr(self.rotor_loads_i, att, val_i)

            # hub_loads
            for att in self.hub_loads.list_vars():
                if att[0] is '_':
                    continue
                if att is 'wsp':
                    continue
                val = getattr(self.hub_loads, att)
                if not val.tolist():
                    continue
                val_i = np.zeros([n])

                val_i[:self.N] = pchip(self.rotor_loads.wsp[:i_max_ws],
                                       val[:i_max_ws])(ws_i)

                for i, v in enumerate(val[i_max_ws:]):
                    val_i[self.N+i] = v
                setattr(self.hub_loads_i, att, val_i)

            # oper
            for att in self.oper.list_vars():
                if att[0] is '_':
                    continue
                val = getattr(self.oper, att)
                if not val.tolist():
                    continue
                val_i = np.zeros([n])

                val_i[:self.N] = pchip(self.rotor_loads.wsp[:i_max_ws],
                                       val[:i_max_ws])(ws_i)
                for i, v in enumerate(val[i_max_ws:]):
                    val_i[self.N+i] = v
                setattr(self.oper_i, att, val_i)

            # blade_disp
            self.blade_disps_i.tip_pos = np.zeros([n, 3])
            for i in [0, 2]:
                self.blade_disps_i.tip_pos[:self.N, i] =\
                    pchip(self.oper.wsp[:i_max_ws],
                          self.blade_disps.tip_pos[:i_max_ws, i])(ws_i)

            self.blade_disps_i.tip_pos[:self.N, 1] =\
                sharp_curves_interpolation(self.oper.wsp[:i_max_ws],
                                           self.blade_disps.tip_pos[:i_max_ws, 1],
                                           ws_i, ws_r)
            for i in range(3):
                for j, v in enumerate(self.blade_disps.tip_pos[i_max_ws:, i]):
                    self.blade_disps_i.tip_pos[self.N+j, i] = v

            self.blade_disps_i.tip_rot = np.zeros([n])
            self.blade_disps_i.tip_rot[:self.N] = pchip(
                self.oper.wsp[:i_max_ws],
                self.blade_disps.tip_rot[:i_max_ws])(ws_i)

            for j, v in enumerate(self.blade_disps.tip_rot[i_max_ws:]):
                self.blade_disps_i.tip_rot[self.N+j] = v


def stfile2beamvt(filename):
    """read HAWC2 st file and return list of BeamStructureVT's"""

    from fusedwind.turbine.structure_vt import BeamStructureVT  # FIXME:
    sts = []
    stdic = read_hawc2_st_file(filename)
    for stset in stdic:
        st = BeamStructureVT()
        for k, w in stset.iteritems():
            fused_name = k
            if k == 'K':
                fused_name = 'G'
            try:
                setattr(st, fused_name, w)
            except:
                print 'key error', k
        sts.append(st)

    return sts
    
    
class ComputeLoads(object):
    """
    Compute extreme loads based on steady state HAWC2S simulations

    The blade_loads cases do not include contributions from gravity loads.
    These are therefore added manually to the loads.
    Likewise, the blade loads do not include centrifugal forces, also added manually.
    todo: calculate torsional moment
    """
    def __init__(self):

        self.oper = RotorOperationalDataArray()
        self.beam_structure = BeamStructureVT()
        self.pf = BladePlanformVT()
        self.blade_loads = DistributedLoadsArrayVT()
        self.factor = 1.35
        self.g = 9.81
        self.hub_radius = np.array([0.0])
        self.x = np.array([0.0])

        self.lc = LoadVectorCaseList

    def execute(self):

        self.lcs = []
        self.lc = []
        for j, lname in enumerate(self.blade_loads.loads_array):
            load = getattr(self.blade_loads, lname)
            lc = LoadVectorArray()
            for name in lc.list_vars():
                var = getattr(lc, name)
                if isinstance(var, np.ndarray):
                    setattr(lc, name, np.zeros(load.s.shape[0]))
            lc.s = load.s
            r = load.s * self.pf.blade_length + self.hub_radius
            lc.case_id = lname
            lc.Fxm = np.zeros(load.s.shape[0])
            lc.Fym = np.zeros(load.s.shape[0])
            # rotate to local profile coordinate system
            pitch = self.oper.pitch[j] * np.pi / 180.
            vhub = self.oper.vhub
            theta = np.interp(load.s, self.pf.s, self.pf.rot_z) * np.pi / 180.
            Fx =  load.Ft * np.cos(theta + pitch) + load.Fn * np.sin(theta + pitch)
            Fy =  load.Fn * np.cos(theta + pitch) - load.Ft * np.sin(theta + pitch)

            # compute contribution from mass
            dm = np.interp(load.s * self.pf.smax * self.pf.blade_length, self.beam_structure.s, self.beam_structure.dm)
            # Fmass = np.array([np.trapz(dm[i:], load.s[i:]) for i in range(load.s.shape[0])]) * 9.81
            Fmass = dm * self.g
            Fxmass = Fmass * np.cos(theta + pitch)
            Fymass = Fmass * np.sin(theta + pitch)

            # centrifugal acceleration a = (omega * r)**2 / r + g
            acc = (self.oper.rpm[j] * 2 * np.pi / 60.)**2 * r + self.g
            lc.acc = acc
            if vhub > 25.:
                factor = 1.35
            else:
                factor = 1.1

            for i in range(load.s.shape[0]):
                lc.Fx[i] = (np.trapz(Fx[i:] + Fxmass[i:], r[i:])) * factor
                lc.Fy[i] = (np.trapz(Fy[i:] + Fymass[i:], r[i:])) * factor
                lc.Fz[i] = np.trapz(acc[i:] * dm[i:], r[i:]) * factor
                lc.Mx[i] = -np.trapz((Fy[i:] + Fxmass[i:]) * (r[i:] - r[i]), r[i:]) * factor
                lc.My[i] = np.trapz((Fx[i:] + Fymass[i:]) * (r[i:] - r[i]), r[i:]) * factor

            self.lcs.append(lc.copy())

        for j, x in enumerate(self.x):
            lc = LoadVectorCaseList()
            lc.s = x
            for name in ['Fx', 'Fy', 'Fz', 'Mx', 'My']:
                # positive components
                lv = LoadVector()
                for case in self.lcs:
                    c = case._interp_s(x)
                    if getattr(c, name) > getattr(lv, name):
                        lv = c.copy()
                lc.cases.append(lv.copy())

                # negative components
                lv = LoadVector()
                for case in self.lcs:
                    c = case._interp_s(x)
                    if getattr(c, name) < getattr(lv, name):
                        lv = c.copy()
                lc.cases.append(lv.copy())
            self.lc.append(lc.copy())