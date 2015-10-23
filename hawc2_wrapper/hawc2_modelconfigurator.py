


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