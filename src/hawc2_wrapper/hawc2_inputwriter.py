import os
import numpy as np
from hawc2_vartrees import HAWC2VarTrees


def _get_fmt(val):

    def _get_fmt1(val):
        if isinstance(val, str):
            return '%s'
        elif isinstance(val, int):
            return '%i'
        elif isinstance(val, float):
            return '%24.15f'

    if isinstance(val, list):
        return ' '.join([_get_fmt1(v) for v in val])
    else:
        return _get_fmt1(val)


def write_pcfile(path, pc):
    """write the blade airfoils"""

    fid = open(path, 'w')
    fid.write('%i %s\n' % (pc.nset, pc.desc))
    for i in range(pc.nset):
        pcset = pc.pc_sets[i]
        fid.write('%i\n' % pcset.np)
        for j in range(pcset.np):
            polar = pcset.polars[j]
            fid.write('%i %i %f %s\n' %
                      (j + 1, polar.aoa.shape[0], polar.rthick, polar.desc))
            for k in range(polar.aoa.shape[0]):
                fid.write((4*'%24.15e '+'\n') % (polar.aoa[k], polar.cl[k],
                                                 polar.cd[k], polar.cm[k]))
    fid.close()


def write_aefile(path, b):
    """write the blade shape to the ae_filename"""

    fid = open(path, 'w')

    fid.write('1 HAWC_AE data\n')
    fid.write("1 %i r,c,t/c prin.set\n" % (b.s.shape[0]))
    data = np.array([b.s,
                     b.chord,
                     np.minimum(100., b.rthick),
                     b.aeset]).T
    np.savetxt(fid, data, fmt="%.20e %.20e %.20e %i")
    fid.close()


def write_stfile(path, body, case_id):

    """write the beam structural data to an st_filename"""
    if body.st_input_type is 0:
        header = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh',
                  'E', 'G', 'I_x', 'I_y', 'K', 'k_x', 'k_y', 'A', 'pitch',
                  'x_e', 'y_e']
        # for readable files with headers above the actual data column
        exp_prec = 20             # exponential precesion
        col_width = exp_prec + 8  # column width required for exp precision
        header_full = '='*20*col_width + '\n'
        header_full += ''.join([(hh + ' [%i]').center(col_width + 1) % i
                               for i, hh in enumerate(header)])+'\n'
        header_full += '='*20*col_width + '\n'

    else:
        header = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'pitch', 'x_e',
                  'y_e', 'K_11', 'K_12', 'K_13', 'K_14', 'K_15', 'K_16',
                  'K_22', 'K_23', 'K_24', 'K_25', 'K_26', 'K_33', 'K_34',
                  'K_35', 'K_36', 'K_44', 'K_45', 'K_46', 'K_55', 'K_56',
                  'K_66']
        # for readable files with headers above the actual data column
        exp_prec = 20             # exponential precesion
        col_width = exp_prec + 8  # column width required for exp precision
        header_full = '='*32*col_width + '\n'
        header_full += ''.join([(hh + ' [%i]').center(col_width + 1) % i
                                for i, hh in enumerate(header)])+'\n'
        header_full += '='*32*col_width + '\n'

    fid = open(path, 'w')
    fid.write('%i  number of sets, Nset\n' % body.body_set[1])
    fid.write('-----------------\n')
    fid.write('#1 written using the HAWC2 OpenMDAO wrapper\n')
    fid.write('Case ID: %s\n' % case_id)
    if body.st_input_type is 0:
        for i, st in enumerate(body.beam_structure):
            fid.write(header_full)
            fid.write('$%i %i\n' % (i + 1, st.s.shape[0]))
            data = np.array([st.s,
                             st.dm,
                             st.x_cg,
                             st.y_cg,
                             st.ri_x,
                             st.ri_y,
                             st.x_sh,
                             st.y_sh,
                             st.E,
                             st.G,
                             st.I_x,
                             st.I_y,
                             st.K,
                             st.k_x,
                             st.k_y,
                             st.A,
                             st.pitch,
                             st.x_e,
                             st.y_e]).T
            np.savetxt(fid, data, fmt='%'+' %i.%ie' % (col_width, exp_prec))

    else:
        for i, st in enumerate(body.beam_structure):
            fid.write(header_full)
            fid.write('$%i %i\n' % (i + 1, st.s.shape[0]))
            data = np.array([st.s,
                             st.dm,
                             st.x_cg,
                             st.y_cg,
                             st.ri_x,
                             st.ri_y,
                             st.pitch,
                             st.x_e,
                             st.y_e,
                             st.K_11,
                             st.K_12,
                             st.K_13,
                             st.K_14,
                             st.K_15,
                             st.K_16,
                             st.K_22,
                             st.K_23,
                             st.K_24,
                             st.K_25,
                             st.K_26,
                             st.K_33,
                             st.K_34,
                             st.K_35,
                             st.K_36,
                             st.K_44,
                             st.K_45,
                             st.K_46,
                             st.K_55,
                             st.K_56,
                             st.K_66]).T
            np.savetxt(fid, data, fmt='%' + ' %i.%ie' % (col_width, exp_prec))
    fid.close()


class HAWC2InputWriter(object):

    def __init__(self):

        self.from_file = True
        self.case_id = 'hawc2_case'
        self.htc_master = []  # Final master list of inputs

        self.data_directory = 'data'
        self.res_directory = 'res'
        self.turb_directory = 'turb'
        self.log_directory = 'logfile'
        self.control_directory = 'control'

        self.vartrees = HAWC2VarTrees()
        self.Flag = False
        self.force_execute = True

        self.htc_master = []
        self.structure = []
        self.aerodrag = []
        self.controlinp = []
        self.sensors = []

        self._nbodies = 0

        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)

    def write_master(self):

        # print 'writing case id %s' % self.case_id
        fid = open(self.case_id + '.htc', 'w')

        for i, line in enumerate(self.htc_master):
                line = line.rstrip()+'\n'
                self.htc_master[i] = line
                fid.write(line)
        fid.close()

    def write_all(self):

        self.htc_master = []
        self.controlinp = []
        self.sensors = []

        self.configure_wt()
        self.write_simulation()
        self.write_wind()
        self.write_aero()
        self.write_aerodrag()
        self.write_structure_res()
        self.update_c12axis()
        self.write_structure()
        self.write_dlls()
        self.write_output()

        self.htc_master.extend(self.structure)
        self.htc_master.extend(self.controlinp)
        self.htc_master.extend(self.sensors)

    def execute(self):

        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)
        self.case_idout = self.case_id
        self.write_all()
        self.write_master()
        self.write_pcfile()
        self.write_aefile()
        self.Flag = True

    def configure_wt(self):

        if not self.from_file:
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

    def write_simulation(self):
        """ write simulation block """

        sim = []
        sim.append('begin simulation')
        sim.append('time_stop    %s' % self.vartrees.sim.time_stop)
        sim.append('solvertype   %i' %
                   self.vartrees.sim.solvertype)
        sim.append('convergence_limits %9.2e %9.2e %9.2e' %
                   tuple(self.vartrees.sim.convergence_limits))
        sim.append('on_no_convergence continue')
        sim.append('max_iterations %i' % self.vartrees.sim.max_iterations)
        sim.append('logfile %s' %
                   (os.path.join(self.log_directory, self.case_id + '.log')))
        sim.append('begin newmark')
        sim.append('  deltat    %1.3f' % self.vartrees.sim.newmark_deltat)
        sim.append('end newmark')
        sim.append('end simulation')
        # Adding indent and semicolons
        sim[0] += ';'
        sim[-1] += ';'
        for iw, w in enumerate(sim[1:-1]):
            sim[iw+1] = '  ' + w + ';'
        self.htc_master.extend(sim)

    def write_wind(self):
        """ write wind definition block """

        wind_vt = self.vartrees.wind
        fmt = ' %12.6e'
        wind = []
        wind.append('begin wind')
        wind.append('density' + fmt % wind_vt.density)
        wind.append('wsp' + fmt % wind_vt.wsp)
        wind.append('tint' + fmt % wind_vt.tint)
        wind.append('horizontal_input  %i' % wind_vt.horizontal_input)
        wind.append('windfield_rotations' + 3*fmt %
                    tuple(wind_vt.windfield_rotations))
        wind.append('center_pos0' + 3*fmt % tuple(wind_vt.center_pos0))
        wind.append('shear_format         %i' % wind_vt.shear_type +
                    fmt % wind_vt.shear_factor)
        wind.append('turb_format          %i' % wind_vt.turb_format)
        wind.append('tower_shadow_method  %i' % wind_vt.tower_shadow_method)

        for wind_ramp in wind_vt.wind_ramp_abs:
            wind.append('wind_ramp_abs' + 4*fmt % tuple(wind_ramp))
        if self.vartrees.wind.scale_time_start > 0:
            wind.append('scale_time_start' + fmt % wind_vt.scale_time_start)
        if self.vartrees.wind.wind_ramp_t1 > 0:
            wind.append('wind_ramp_factor' + 4*fmt %
                        (wind_vt.wind_ramp_t0, wind_vt.wind_ramp_t1,
                         wind_vt.wind_ramp_factor0, wind_vt.wind_ramp_factor1))
        if wind_vt.iec_gust:
            wind.append('iec_gust %s' % wind_vt.iec_gust_type + 4*fmt %
                        (wind_vt.G_A, wind_vt.G_phi0, wind_vt.G_t0,
                         wind_vt.G_T))

        if wind_vt.turb_format == 1:
            wind.extend(self.write_mann_turbulence())

        if wind_vt.tower_shadow_method > 0:
            wind.extend(self.write_tower_potential())

        wind.append('end wind')
        # Adding indent and semicolons
        wind[0] += ';'
        wind[-1] += ';'
        for iw, w in enumerate(wind[1:-1]):
            wind[iw+1] = '  ' + w + ';'
        self.htc_master.extend(wind)

    def write_mann_turbulence(self):
        """ write mann turbulence model"""

        fmt = ' %12.6e'
        mann_vt = self.vartrees.wind.mann
        mann = []
        mann.append('begin mann')
        if mann_vt.create_turb:
            mann.append(('create_turb_parameters' + 3*fmt + ' %i' + fmt) %
                        (mann_vt.L, mann_vt.alfaeps, mann_vt.gamma,
                         mann_vt.seed, mann_vt.highfrq_compensation))
        mann.append('filename_u %s' % (os.path.join(self.turb_directory,
                                       mann_vt.turb_base_name + '_u.bin')))
        mann.append('filename_v %s' % (os.path.join(self.turb_directory,
                                       mann_vt.turb_base_name + '_v.bin')))
        mann.append('filename_w %s' % (os.path.join(self.turb_directory,
                                       mann_vt.turb_base_name + '_w.bin')))
        mann.append(('box_dim_u %5i'+fmt) % (mann_vt.box_nu, mann_vt.box_du))
        mann.append(('box_dim_v %5i'+fmt) % (mann_vt.box_nv, mann_vt.box_dv))
        mann.append(('box_dim_w %5i'+fmt) % (mann_vt.box_nw, mann_vt.box_dw))
        mann.append('std_scaling'+3*fmt % tuple(mann_vt.std_scaling))
        mann.append('end mann')

        for im, m in enumerate(mann[1:-1]):
            mann[im+1] = '  ' + m
        return mann

    def write_tower_potential(self):
        """ write tower shadow with potential method"""

        fmt = ' %12.6e'
        if hasattr(self.vartrees.wind, 'tower_potential'):
            tp = self.vartrees.wind.tower_potential
            tower_pot = []
            tower_pot.append('begin tower_shadow_potential_2')
            tower_pot.append('tower_mbdy_link %s' % tp.tower_mbdy_link)
            tower_pot.append('nsec %d' % tp.nsec)
            for sec in tp.sections:
                tower_pot.append('radius' + 2*fmt % tuple(sec))
            tower_pot.append('end tower_shadow_potential_2')

        for im, m in enumerate(tower_pot[1:-1]):
            tower_pot[im+1] = '  ' + m
        return tower_pot

    def write_aero(self):
        """ write aerodynamic block """

        aerovt = self.vartrees.aero
        aero = []
        aero.append('begin aero')
        aero.append('nblades  %d' % aerovt.nblades)
        aero.append('hub_vec %s -3' % 'shaft')
        for link in self.vartrees.aero.links:
            aero.append('link %i mbdy_c2_def %s' % (link[0], link[2]))
        aero.append('ae_filename ./%s/%s_ae.dat' % (self.data_directory,
                                                    self.case_id))
        aero.append('pc_filename ./%s/%s_pc.dat' % (self.data_directory,
                                                    self.case_id))
        aero.append('induction_method %i' % aerovt.induction_method)
        aero.append('aerocalc_method  %i' % aerovt.aerocalc_method)
        if aerovt.aero_distribution_file != '':
            aero.append('aero_distribution %s %i' %
                        (aerovt.aero_distribution_file,
                         aerovt.aero_distribution_set))
        else:
            aero.append('aerosections %i' % aerovt.aerosections)
        aero.append('ae_sets         %s' % ' '.join(map(str, aerovt.ae_sets)))
        aero.append('tiploss_method  %i' % aerovt.tiploss_method)
        aero.append('dynstall_method %i' % aerovt.dynstall_method)
        aero.append('end aero')

        for ia, a in enumerate(aero):
            aero[ia] = a + ';'
        for ia, a in enumerate(aero[1:-1]):
            aero[ia+1] = '  ' + a

        self.htc_master.extend(aero)

    def write_aerodrag(self):
        """ write aerodrag block """

        aerodrag = []
        fmt = ' %12.6e'
        if len(self.vartrees.aerodrag.elements) > 0:
            aerodrag.append('begin aerodrag')
            for i, e in enumerate(self.vartrees.aerodrag.elements):
                aerodrag.append('begin aerodrag_element')
                aerodrag.append('  mbdy_name %s' % e.mbdy_name)
                aerodrag.append('  aerodrag_sections %s %i' % (
                                     e.dist, e.calculation_points))
                aerodrag.append('  nsec %d' % e.nsec)
                for j in range(e.nsec):
                    aerodrag.append('  sec' + 3*fmt % tuple(e.sections[j][:]))
                aerodrag.append('end aerodrag_element')
            aerodrag.append('end aerodrag')
            for ia, a in enumerate(aerodrag):
                aerodrag[ia] = a + ';'
            for ia, a in enumerate(aerodrag[1:-1]):
                aerodrag[ia+1] = '  ' + a
            self.htc_master.extend(aerodrag)

    def write_structure_res(self):
        """ write result files for eigen analysis """

        structure_out = []
        if self.vartrees.sim.eig_out:
            structure_out.append('beam_output_file_name ./info/%s_beam.dat;' %
                                 self.case_id)
            structure_out.append('body_output_file_name ./info/%s_body.dat;' %
                                 self.case_id)
            structure_out.append('struct_inertia_output_file_name \
                                 ./info/%s_inertia.dat;' % self.case_id)
            structure_out.append('body_eigenanalysis_file_name \
                                 ./info/%s_body_eigs.dat;' % self.case_id)
            structure_out.append('constraint_output_file_name \
                                 ./info/%s_constraints.dat;' % self.case_id)
            structure_out.append('structure_eigenanalysis_file_name \
                                 ./info/%s_struct_eigs.dat 1;' % self.case_id)

        self.structure.extend(structure_out)

    def write_main_bodies(self):
        """ write all main bodies """

        for name in self.vartrees.body_order:
            self.structure.extend(self.write_main_body(name))

    def write_main_body(self, body_name):
        """ write one main body """

        fmt = ' %12.6e'
        body = self.vartrees.main_bodies.get_main_body(body_name)
        main_body = []
        main_body.append('begin main_body')
        if body.copy_main_body is not '':
            main_body.append('name           %s' % body.body_name)
            main_body.append('copy_main_body %s' % body.copy_main_body)
        else:
            main_body.append('name        %s' % body.body_name)
            main_body.append('type        timoschenko')
            main_body.append('nbodies     %d' % body.nbodies)
            main_body.append('node_distribution     c2_def')

            if body.damping_type is 'ani':
                main_body.append('damping_aniso' + 6*fmt %
                                 tuple(body.damping_aniso))
            else:
                main_body.append('damping_posdef' + 6*fmt %
                                 tuple(body.damping_posdef))

            for i in range(len(body.concentrated_mass)):
                main_body.append('concentrated_mass' + 8*fmt %
                                 tuple(body.concentrated_mass[i]))
            main_body.append('begin timoschenko_input')
            tmpname = ''.join([i for i in body.body_name if not i.isdigit()])
            main_body.append('  filename %s' %
                             (os.path.join(self.data_directory, self.case_id +
                                           '_' + tmpname + '_st.dat')))
            if body.st_input_type is not 0:
                main_body.append('  FPM %d' % body.st_input_type)
            main_body.append('  set %d %d' % tuple(body.body_set))
            main_body.append('end timoschenko_input')
            main_body.append('begin c2_def')
            main_body.append('  nsec %i' % body.c12axis.shape[0])
            for i in range(body.c12axis.shape[0]):
                main_body.append('  sec %2i' % (i+1) + 4*' %22.15e' %
                                 tuple(body.c12axis[i, :]))
            main_body.append('end c2_def')
            if len(body.beam_structure) > 0:
                self.write_stfile(body)

        main_body.append('end main_body')

        # add indent and semicolon
        main_body[0] = '  ' + main_body[0] + ';'
        main_body[-1] = '  ' + main_body[-1] + ';'
        for ib, b in enumerate(main_body[1:-1]):
            main_body[ib+1] = '    ' + b + ';'

        return main_body

    def write_orientations(self):

        orientations = []
        orientations.append('begin orientation')
        fmt = ' %12.6e'
        for name in self.vartrees.body_order:
            body = self.vartrees.main_bodies.get_main_body(name)
            for orien in body.orientations:
                if orien.type == 'base':
                    orientations.append('begin base')
                    orientations.append('  body %s' % body.body_name)
                    orientations.append('  inipos' + 3*fmt %
                                        tuple(orien.inipos))
                    for eulerang in orien.body_eulerang:
                        orientations.append('  body_eulerang'+3*fmt %
                                            tuple(eulerang))
                    orientations.append('end base')

                elif orien.type == 'relative':
                    orientations.append('begin relative')
                    fmt2 = '  body1 '+_get_fmt(orien.body1)
                    orientations.append(fmt2 % tuple(orien.body1))
                    fmt2 = '  body2 '+_get_fmt(orien.body2)
                    orientations.append(fmt2 % tuple(orien.body2))
                    for eulerang in orien.body2_eulerang:
                        orientations.append('  body2_eulerang'+3*fmt %
                                            tuple(eulerang))
                    if orien.mbdy2_ini_rotvec_d1[3] != 0.:
                        orientations.append('  body2_ini_rotvec_d1'+4*fmt %
                                            tuple(orien.mbdy2_ini_rotvec_d1))
                    orientations.append('end relative')
        orientations.append('end orientation')
        # add indent and semicolon
        for io, o in enumerate(orientations):
            orientations[io] = '  ' + o + ';'
        for io, o in enumerate(orientations[1:-1]):
            orientations[io+1] = '  ' + o
        self.structure.extend(orientations)

    def write_constraints(self):

        constraints = []
        fmt = ' %12.6e'
        constraints.append('begin constraint')
        for name in self.vartrees.body_order:
            body = self.vartrees.main_bodies.get_main_body(name)
            for con in body.constraints:
                if con.con_type in ['fix0', 'fix2', 'fix3']:
                    constraints.append('begin %s' % con.con_type)
                    constraints.append('  body %s' % con.mbdy)
                    if con.disable_at > 0.:
                        constraints.append('  disable_at %s' %
                                           con.disable_at)
                    if con.con_type in ['fix2', 'fix3']:
                        constraints.append('  dof %i %i %i' % tuple(con.dof))
                    constraints.append('end %s' % con.con_type)

                elif con.con_type in ['fix1', 'fix4']:
                    constraints.append('begin %s' % con.con_type)
                    fmt2 = '  body1 ' + _get_fmt(con.mbdy1)
                    constraints.append(fmt2 % tuple(con.mbdy1))
                    fmt2 = '  body2 ' + _get_fmt(con.mbdy2)
                    constraints.append(fmt2 % tuple(con.mbdy2))
                    constraints.append('end %s' % con.con_type)

                elif 'bearing' in con.con_type:
                    constraints.append('begin %s' % con.con_type)
                    constraints.append('  name %s' % con.name)
                    fmt2 = '  body1 '+_get_fmt(con.mbdy1)
                    constraints.append(fmt2 % tuple(con.mbdy1))
                    fmt2 = '  body2 '+_get_fmt(con.mbdy2)
                    constraints.append(fmt2 % tuple(con.mbdy2))
                    constraints.append(('  bearing_vector %i'+3*fmt) %
                                       tuple(con.bearing_vector))
                    if con.con_type == 'bearing3':
                        constraints.append('  omegas'+3*fmt % con.omegas)
                    else:
                        if con.disable_at > 0:
                            constraints.append('  disable_at %s' %
                                               con.disable_at)
                    constraints.append('end %s' % con.con_type)

        constraints.append('end constraint')
        # add indent and semicolon
        for ic, c in enumerate(constraints):
            constraints[ic] = '  ' + c + ';'
        for ic, c in enumerate(constraints[1:-1]):
            constraints[ic+1] = '  ' + c
        self.structure.extend(constraints)

    def write_structure(self):
        """ write model structure"""

        self.structure = []
        self.structure.append('begin new_htc_structure;')
        self.write_main_bodies()
        self.write_orientations()
        self.write_constraints()
        self.structure.append('end new_htc_structure;')

    def write_dlls(self):
        """
        """
        self.controlinp.append('begin dll;')
        for name in self.vartrees.dlls_order:
            self.controlinp.extend(self.write_dll(name))
        self.controlinp.append('end dll;')
        for i, c in enumerate(self.controlinp[1:-1]):
            self.controlinp[i+1] = '  ' + c

    def write_dll(self, dll_name):
        """ write general type2 dll"""

        fmt = '%2i  %12.6e'
        dll = []
        dll_vt = getattr(self.vartrees.dlls, dll_name)
        dll_init = dll_vt. dll_init
        dll.append('begin type2_dll')
        dll.append('name %s' % dll_vt.name)
        dll.append('filename %s' % dll_vt.filename)
        dll.append('dll_subroutine_init %s' % dll_vt.dll_subroutine_init)
        dll.append('dll_subroutine_update %s' % dll_vt.dll_subroutine_update)
        dll.append('arraysizes_init %i %i' % tuple(dll_vt.arraysizes_init))
        dll.append('arraysizes_update %i %i' % tuple(dll_vt.arraysizes_update))
        dll.append('begin init')

        for i in range(len(dll_init.init_dic.keys())):
            val = getattr(dll_init,
                          dll_init.init_dic[i+1][0])/dll_init.init_dic[i+1][1]
            dll.append('  constant ' + fmt % (i+1, val))

        dll.append('end init')

        dll.append('begin output')
        for i in range(len(dll_vt.output.out_dic.keys())):
            val = getattr(dll_vt.output, 'out_%i' % (i + 1))
            dll.append('  %s' % val._print())
        dll.append('end output')

        if len(dll_vt.actions.action_dic.keys()) > 0:
            dll.append('begin actions')
            for i in range(len(dll_vt.actions.action_dic.keys())):
                val = getattr(dll_vt.actions, 'action_%i' % (i + 1))
                dll.append('  %s' % val._print())
            dll.append('end actions')

        dll.append('end type2_dll')

        # add indent and semicolon
        dll[0] += ';'
        dll[-1] += ';'
        for i, d in enumerate(dll[1:-1]):
            dll[i+1] = '  ' + d + ';'

        return dll

    def update_c12axis(self):

        self.vartrees.main_bodies.blade1.c12axis = \
            self.vartrees.blade_ae.c12axis.copy()
        self.vartrees.main_bodies.blade1.beam_structure = \
            self.vartrees.blade_structure

    def write_output(self):

        sns = []
        sns.append('begin output')
        sns.append('  filename %s' % (os.path.join(self.res_directory,
                                                   self.case_id)))
        sns.append('  time %3.6f %3.6f' % (self.vartrees.output.time_start,
                                           self.vartrees.sim.time_stop))
        sns.append('  data_format %s' % self.vartrees.output.out_format)
        sns.append('  buffer 1')

        for i in range(len(self.vartrees.output.sensor_list)):
            sns.append('  ' + self.vartrees.output.sensor_list[i])
        sns.append('end output')

        sns = [s+';' for s in sns]
        self.sensors.extend(sns)

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

    def write_aefile(self):

        path = os.path.join(self.data_directory, self.case_id + '_ae.dat')
        write_aefile(path, self.vartrees.blade_ae)

    def write_stfile(self, body):

        tmpname = ''.join([i for i in body.body_name if not i.isdigit()])
        path = os.path.join(self.data_directory,
                            self.case_id + '_' + tmpname + '_st.dat')
        write_stfile(path, body, self.case_id)

    def write_pcfile(self):

        path = os.path.join(self.data_directory, self.case_id + '_pc.dat')
        write_pcfile(path, self.vartrees.airfoildata)


class HAWC2AeroInputWriter(HAWC2InputWriter):

    def write_main_bodies(self):
        """
        specific writer of the HAWC2aero blade c2_axis
        """

        pass
        # fid = open(path, 'w')
        # lines = []
        # lines.append('begin blade_c2_def;')
        # lines.append('    blade 0 %i;' % b.main_axis.shape[0])
        # for i in range(b.main_axis.shape[0]):
        #     lines.append('        sec %i  %f %f %f %f;' % (i + 1,
        #                                                     c12axis[i, 0],
        #                                                     c12axis[i, 1],
        #                                                     c12axis[i, 2],
        #                                                     c12axis[i, 3]))
        # lines.append('end blade_c2_def;')
        # for line in lines:
        #     self.master.append(line+'')
        # fid.close()


class HAWC2SInputWriter(HAWC2InputWriter):

    def __init__(self):
        super(HAWC2SInputWriter, self).__init__()
        self.h2s = []
        self.set_tsr_flag = False  # Manually set omega according to TSR
        self.wsp_cases = []  # Wind speeds for which to run HAWC2S
        self.cases = []  # Other cases to run

    def write_all(self):
        self.configure_wt()
        self.write_aero()

        turb_format = self.vartrees.wind.turb_format
        self.vartrees.wind.turb_format = 0
        self.write_wind()
        self.vartrees.wind.turb_format = turb_format

        self.write_structure_res()
        self.update_c12axis()
        self.write_main_bodies()
        self.write_orientations()
        self.write_constraints()
        self.structure.insert(0, 'begin new_htc_structure;')
        self.structure.append('end new_htc_structure;')

        self.htc_master.extend(self.structure)
        self.write_hawcstab2()

    def execute(self):

        self.h2s = []
        self.htc_master = []
        self.structure = []
        self.aerodrag = []
        self.controlinp = []
        self.sensors = []

        if not os.path.exists(self.data_directory):
            os.mkdir('data')
        self.case_idout = self.case_id

        self.write_all()
        self.write_master()
        self.write_pcfile()
        self.write_aefile()

    def write_hawcstab2(self):

        self.h2s.append('begin hawcstab2;')
        self.write_hawcstab2_structure()
        self.write_operational_data_file()
        self.write_h2s_operational_data()
        self.write_h2s_control()
        self.write_h2s_commands()
        self.h2s.append('end hawcstab2;')
        self.htc_master.extend(self.h2s)

    def write_h2s_commands(self):

        opt_vt = self.vartrees.h2s.options
        cmd = []
        for name in self.vartrees.h2s.commands:
            if name == 'compute_optimal_pitch_angle':
                cmd.append('compute_optimal_pitch_angle use_operational_data')

            elif name == 'compute_steady_states':
                cmd.append('compute_steady_states %s %s %s %s' %
                           (opt_vt.bladedeform, opt_vt.tipcorrect,
                            opt_vt.induction, opt_vt.gradients))

            elif name == 'compute_steadystate':
                cmd.append('compute_steadystate %s %s %s %s' %
                           (opt_vt.bladedeform, opt_vt.tipcorrect,
                            opt_vt.induction, opt_vt.gradients))

            elif name == 'compute_stability_analysis':
                cmd.append(('compute_stability_analysis %s %s %i'+4*'%12.6e' +
                           ' %s') % (opt_vt.matrixwriteout,
                                     opt_vt.eigenvaluewriteout,
                                     opt_vt.number_of_modes,
                                     opt_vt.maximum_damping,
                                     opt_vt.minimum_frequency,
                                     opt_vt.zero_pole_threshold,
                                     opt_vt.aero_deflect_ratio,
                                     opt_vt.frequencysorting))

            elif name == 'compute_aeroservoelastic':
                cmd.append(('compute_aeroservoelastic %s %s %i'+4*'%12.6e'+'%s'
                            ) % (opt_vt.matrixwriteout,
                                 opt_vt.eigenvaluewriteout,
                                 opt_vt.number_of_modes,
                                 opt_vt.maximum_damping,
                                 opt_vt.minimum_frequency,
                                 opt_vt.zero_pole_threshold,
                                 opt_vt.aero_deflect_ratio,
                                 opt_vt.frequencysorting))

            elif name == 'save_cl_matrices_all':
                if opt_vt.vloc_out:
                    cmd.append('save_cl_matrices_all vloc_out')
                else:
                    cmd.append('save_cl_matrices_all')
            elif name == 'compute_structural_modal_analysis':
                if opt_vt.blade_only:
                    cmd.append('compute_structural_modal_analysis '
                               'bladeonly %i' % opt_vt.number_of_modes)
                else:
                    cmd.append('compute_structural_modal_analysis '
                               'nobladeonly %i' % opt_vt.number_of_modes)

            elif name == 'basic_dtu_we_controller':
                init = self.vartrees.dlls.risoe_controller.dll_init

                cmd.append('basic_dtu_we_controller %1.9e %1.9e %1.9e '
                           '%1.9e %1.9e %1.9e %1.9e %1.9e %1.9e %1.9e '
                           '%i %1.9e %1.9e %1.9e' %
                           (init.pgTorque, init.igTorque, init.Qg,
                            init.pgPitch, init.igPitch,
                            init.KK1, init.KK2,
                            init.generatorFreq, init.generatorDamping,
                            init.ffFreq, init.generatorSwitch,
                            init.Kp2, init.Ko1, init.Ko2))
            else:
                cmd.append(name)
        cmd = ['  ' + c + ';' for c in cmd]
        self.h2s.extend(cmd)

    def write_operational_data_file(self):

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

    def write_hawcstab2_structure(self):

        h2s_vt = self.vartrees.h2s
        st = []
        st.append('begin ground_fixed_substructure')
        for name in h2s_vt.ground_fixed.main_body:
            st.append('  main_body %s' % name)
        if h2s_vt.ground_fixed.log_decrements[0] != 0:
            st.append('  log_decrements %12.6f %12.6f' %
                      tuple(h2s_vt.ground_fixed.log_decrements))
        st.append('end ground_fixed_substructure')

        st.append('begin rotating_axissym_substructure')
        for name in h2s_vt.rotating_axissym.main_body:
            st.append('  main_body %s' % name)
        if h2s_vt.rotating_axissym.log_decrements[0] != 0:
            st.append('  log_decrements %12.6f %12.6f' %
                      tuple(h2s_vt.rotating_axissym.log_decrements))
        st.append('end rotating_axissym_substructure')

        st.append('begin rotating_threebladed_substructure')
        for name in h2s_vt.rotating_threebladed.main_body:
            st.append('  main_body %s' % name)
        if h2s_vt.rotating_threebladed.log_decrements[0] != 0:
            st.append('  log_decrements' + 6*' %12.6f' %
                      tuple(h2s_vt.rotating_threebladed.log_decrements))
        st.append('  second_order_actuator %s %12.6e %12.6e' %
                  (h2s_vt.second_order_actuator.name,
                   h2s_vt.second_order_actuator.frequency,
                   h2s_vt.second_order_actuator.damping))
        st.append('end rotating_threebladed_substructure')

        st = ['  ' + s + ';' for s in st]
        self.h2s.extend(st)

    def write_h2s_control(self):

        ctr = []
        dll_init = self.vartrees.dlls.risoe_controller.dll_init
        ctr.append('begin controller_tuning')
        ctr.append('  partial_load %3.6f %3.6f' % (dll_init.poleFreqTorque,
                                                   dll_init.poleDampTorque))
        ctr.append('  full_load %3.6f %3.6f' % (dll_init.poleFreqPitch,
                                                dll_init.poleDampPitch))
        ctr.append('  gain_scheduling %d' % dll_init.gainScheduling)
        ctr.append('  constant_power %d' % dll_init.generatorSwitch)

        if len(self.vartrees.h2s.options.regions) > 1:
            ctr.append('  regions %i %i %i %i' %
                       tuple(self.vartrees.h2s.options.regions))
        ctr.append('end controller_tuning')

        ctr.append('begin controller')
        ctr.append('  begin input')
        for i in range(len(self.vartrees.h2s.ch_list_in.sensor_list)):
            ctr.append('    '+self.vartrees.h2s.ch_list_in.sensor_list[i])
        ctr.append('  end input')
        ctr.append('  begin output')
        for i in range(len(self.vartrees.h2s.ch_list_out.sensor_list)):
            ctr.append('    '+self.vartrees.h2s.ch_list_out.sensor_list[i])
        ctr.append('  end output')

        ctr.append('end controller')

        ctr = ['  ' + c + ';' for c in ctr]
        self.h2s.extend(ctr)

    def write_h2s_operational_data(self):

        dll_init = self.vartrees.dlls.risoe_controller.dll_init
        opt = []
        opt.append('operational_data_filename %s' % self.case_id + '.opt')

        opt.append('begin operational_data')
        opt.append('  windspeed %12.6e %12.6e %d' % (dll_init.Vin, dll_init.Vout,
                                                     dll_init.nV))
        opt.append('  genspeed %12.6e %12.6e' % (dll_init.minRPM,
                                                 dll_init.maxRPM))
        opt.append('  gearratio %12.6e' % dll_init.gearRatio)
        opt.append('  minpitch %12.6e' % dll_init.minPitch)
        opt.append('  opt_lambda %22.15e' % dll_init.designTSR)
        opt.append('  maxpow %12.6e' % dll_init.ratedAeroPower)
        opt.append('  prvs_turbine %d' % dll_init.prvs_turbine)
        opt.append('  include_torsiondeform %d' %
                   self.vartrees.h2s.options.include_torsiondeform)
        if self.vartrees.h2s.options.remove_torque_limits:
            opt.append('  remove_torque_limits %d' %
                       self.vartrees.h2s.options.remove_torque_limits)
        opt.append('end operational_data')

        opt = ['  ' + o + ';' for o in opt]
        self.h2s.extend(opt)

if __name__ == '__main__':

    from hawc2_inputreader import HAWC2InputReader

    a = HAWC2InputReader()
    a.htc_master_file = '.\DTU_10MW_RWT_hs2.htc'#'.\dlc12_wsp04_wdir010_s17001.htc'
    a.execute()

    b = HAWC2SInputWriter()
    b.vartrees = a.vartrees
    b.execute()
