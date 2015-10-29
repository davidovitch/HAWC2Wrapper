"""pure python reader for HAWC2 htc files"""

import numpy as np


def _convert_val(val):
    """
    convert a string value read from a file into either int, float or
    string
    """

    try:
        return int(val)
    except:
        try:
            return float(val)
        except:
            return val


class Entry(object):
    """class for "name: val" entries in input files"""

    def __init__(self, name, val, comment=''):

        self.name = name
        self.val = val
        self.comment = comment

    def _print(self, tab=0):
        """pretty print the entry"""

        if isinstance(self.val, str):
            return '%s%s' % (tab * ' ', self.name + ' ' + self.val)
        try:
            return '%s%s' % (tab * ' ', self.name + ' ' +
                             ' '.join(map(str, self.val)))
        except:
            return '%s%s' % (tab * ' ', self.name + ' ' +
                             str(self.val))


class Section(object):
    """Class for list of Entry objects in a HAWC2 htc file"""

    def __init__(self, name):

        self.name = name
        self.entries = []
        self.comment = ''

    def get_entry(self, name):
        """convenience method to get a specific entry from section"""

        items = []
        for entry in self.entries:
            if entry.name == name:
                if isinstance(entry, Section):
                    return entry
                else:
                    items.append(entry.val)
        if len(items) == 0:
            return None
        if len(items) == 1:
            return items[0]
        else:
            return items

    def _next(self):

        try:
            entry = self.entries[self.pos]
            self.pos += 1
            return entry
        except:
            return False

    def _print(self, tab=0):
        """pretty print section recursively"""

        self.pos = 0
        string = ''

        string += '%sbegin %s' % (tab * ' ', self.name + ' ;\n')
        while self.pos < len(self.entries):
            entry = self._next()
            string += entry._print(tab=tab + 2)
        string += '%send %s' % (tab * ' ', self.name + ' ;\n')

        return string


class HAWC2InputDict(object):
    """
    class for reading a HAWC2 htc file

    file is read into a nested list with Section objects with lists of
    Entry objects with name, val for each input parameter.

    all values are converted to either integer, float or string

    pc, ae, and st files are not read.
    """

    def __init__(self):

        self.body_order = []

    def _print(self):

        string = ''
        for section in self.htc:
            string += section._print()
        return string

    def read(self, filename):

        fid = open(filename, 'r')
        self.lines = fid.readlines()
        self.nl = len(self.lines)
        self.htc = []
        self.pos = 0

        tmp = self._read_section(Section('tmp'))
        self.htc = tmp.entries
        self.inp = tmp

    def _read_section(self, section):

        while self.pos < self.nl:
            line = self._next()
            comment = line[line.find(';')+1:].strip()
            param = line[:line.find(';')]
            param = param.strip().split()
            if len(param) == 0:
                continue
            elif param[0] == ';':
                continue
            elif param[0] == 'begin':
                newsection = Section(param[1])
                sec = self._read_section(newsection)
                section.entries.append(sec)
            elif param[0] == 'end':
                return section
            elif param[0] == 'exit':
                return section
            elif param[0] == 'continue_in_file':
                htc = HAWC2InputDict()
                htc.read(param[1])
                self.body_order.extend(htc.body_order)
                section.entries.extend(htc.htc)
            else:
                vals = [_convert_val(val) for val in param[1:]]
                if param[0] == 'name' and section.name == 'main_body':
                    self.body_order.append(param[1])

                if len(vals) == 1:
                    section.entries.append(Entry(param[0], vals[0], comment))
                else:
                    section.entries.append(Entry(param[0], vals, comment))

        return section

    def _next(self):

        try:
            line = self.lines[self.pos]
            self.pos += 1
            return line
        except:
            return False


def read_hawc2_st_file(filename, setnum=-1):
    """
    Reader for a HAWC2 beam structure file.

    The format of the file should be:
    main_s[0] dm[1] x_cg[2] y_cg[3] ri_x[4] ri_y[5] x_sh[6] y_sh[7] E[8] ...
    G[9] I_x[10] I_y[11] K[12] k_x[13] k_y[14] A[15] pitch[16] x_e[17] y_e[18]

    Sub-classes can overwrite this function to change the reader's behaviour.
    """
    right_set = False
    fid = open(filename, 'r')
    st_sets = []
    line = fid.readline()
    while line:
        if (line.find('$') != -1) and right_set:
            ni = int(line.split()[1])
            st_data = np.zeros((ni, 19))
            for i in range(ni):
                tmp = fid.readline().split()
                st_data[i, :] = [float(tmp[j]) for j in range(19)]

            st = {}
            st['s'] = st_data[:, 0]
            st['dm'] = st_data[:, 1]
            st['x_cg'] = st_data[:, 2]
            st['y_cg'] = st_data[:, 3]
            st['ri_x'] = st_data[:, 4]
            st['ri_y'] = st_data[:, 5]
            st['x_sh'] = st_data[:, 6]
            st['y_sh'] = st_data[:, 7]
            st['E'] = st_data[:, 8]
            st['G'] = st_data[:, 9]
            st['I_x'] = st_data[:, 10]
            st['I_y'] = st_data[:, 11]
            st['K'] = st_data[:, 12]
            st['k_x'] = st_data[:, 13]
            st['k_y'] = st_data[:, 14]
            st['A'] = st_data[:, 15]
            st['pitch'] = st_data[:, 16]
            st['x_e'] = st_data[:, 17]
            st['y_e'] = st_data[:, 18]
            st_sets.append(st)

        if line.find('#') != -1:
            if (int(line[1:2]) == setnum) or setnum == -1:
                right_set = True
            else:
                right_set = False
        line = fid.readline()

    fid.close()
    return st_sets


def read_hawc2_stKfull_file(filename, setnum=-1):
    """
    Reader for a HAWC2 beam structure file. Each sectional input is defined
    with 6X6 Constitutive Matrix.

    The format of the file should be:
    main_s[0] dm[1] x_cg[2] y_cg[3] ri_x[4] ri_y[5] x_e[6] y_e[7] pitch[8]
    K_1,1[9] K_1,2[10]  K_1,3[11] K_1,4[12] K_1,5[13] K_1,6[14]
             K_2,2[15]  K_2,3[16] K_2,4[17] K_2,5[18] K_2,6[19]
                        K_3,3[20] K_3,4[21] K_3,5[22] K_3,6[23]
                                  K_4,4[24] K_4,5[25] K_4,6[26]
                                            K_5,5[27] K_5,6[28]
                                                      K_6,6[29]

    First 5 columns are like a standard HAWC2_st input file. The other columns
    define the upper triangular part of the fully populated constitutive
    matrix for each section of the blade.

    Sub-classes can overwrite this function to change the reader's behaviour.
    """
    right_set = False
    fid = open(filename, 'r')
    st_sets = []
    line = fid.readline()
    while line:
        if line.find('$') != -1 and right_set:
            ni = int(line.split()[1])
            st_data = np.zeros((ni, 30))
            for i in range(ni):
                tmp = fid.readline().split()
                st_data[i, :] = [float(tmp[j]) for j in range(30)]

            st = {}
            st['s'] = st_data[:, 0]
            st['dm'] = st_data[:, 1]
            st['x_cg'] = st_data[:, 2]
            st['y_cg'] = st_data[:, 3]
            st['ri_x'] = st_data[:, 4]
            st['ri_y'] = st_data[:, 5]
            st['pitch'] = st_data[:, 6]
            st['x_e'] = st_data[:, 7]
            st['y_e'] = st_data[:, 8]
            st['K_11'] = st_data[:, 9]
            st['K_12'] = st_data[:, 10]
            st['K_13'] = st_data[:, 11]
            st['K_14'] = st_data[:, 12]
            st['K_15'] = st_data[:, 13]
            st['K_16'] = st_data[:, 14]
            st['K_22'] = st_data[:, 15]
            st['K_23'] = st_data[:, 16]
            st['K_24'] = st_data[:, 17]
            st['K_25'] = st_data[:, 18]
            st['K_26'] = st_data[:, 19]
            st['K_33'] = st_data[:, 20]
            st['K_34'] = st_data[:, 21]
            st['K_35'] = st_data[:, 22]
            st['K_36'] = st_data[:, 23]
            st['K_44'] = st_data[:, 24]
            st['K_45'] = st_data[:, 25]
            st['K_46'] = st_data[:, 26]
            st['K_55'] = st_data[:, 27]
            st['K_56'] = st_data[:, 28]
            st['K_66'] = st_data[:, 29]
            st_sets.append(st)
        if line.find('#') != -1:
            if (int(line[1:2]) == setnum) or setnum == -1:
                right_set = True
            else:
                right_set = False
        line = fid.readline()

    fid.close()
    return st_sets


def read_hawc2_pc_file(filename):
    """Read a pc airfoil data file into a dictionary"""

    fid = open(filename, 'r')
    ltmp = fid.readline().strip('\r\n').split()
    nset = int(ltmp[0])
    desc = " ".join(ltmp[1:])
    sets = []
    for i in range(nset):
        pcset = {}
        pcset['polars'] = []
        npo = int(fid.readline().split()[0])
        rawdata = [row.strip().split('\t') for row in fid]
        rthick = []
        ii = 0
        for n in range(npo):
            polar = {}
            line = rawdata[ii][0].split()
            ni = int(line[1])
            polar['np'] = ni
            polar['rthick'] = float(line[2])
            polar['desc'] = " ".join(line[3:])
            data = np.zeros((ni, 4))
            ii += 1
            for i in range(ni):
                dline = rawdata[ii][0]
                data[i, :] = [float(var) for var in dline.split()]
                ii += 1
            polar['aoa'] = data[:, 0]
            polar['cl'] = data[:, 1]
            polar['cd'] = data[:, 2]
            polar['cm'] = data[:, 3]
            rthick.append(polar['rthick'])
            pcset['polars'].append(polar)
        pcset['rthick'] = rthick
        sets.append(pcset)

    fid.close()
    return [desc, sets]


def read_hawc2_ae_file(filename):
    """read blade chord and relative thickness from an ae file"""

    fid = open(filename, 'r')
    blade_ae = {}

    # read header
    line = fid.readline()  # '1 HAWC_AE data\n'
    line = fid.readline()

    # read data
    data = np.loadtxt(fid, usecols=(0, 1, 2, 3))
    blade_ae['s'] = data[:, 0]
    blade_ae['chord'] = data[:, 1]
    blade_ae['rthick'] = data[:, 2]
    blade_ae['aeset'] = data[:, 3]

    fid.close()

    return blade_ae


def read_controller_tuning_file(filename):
    """Read controller tuning output file from HAWCStab2

    Parameters
    ----------

    filename : str
        file name of the controller tuning HS2 output file

    Returns
    -------
        res : dict
            dictionary holding the results with the following keys:
            pi_gen_reg1.K, pi_gen_reg2.Kp, pi_gen_reg2.Ki, pi_gen_reg1.K,
            pi_pitch_reg3.Kp, pi_pitch_reg3.Ki, pi_pitch_reg3.K1,
            pi_pitch_reg3.K2, aero_damp.Kp2, aero_damp.Ko1, aero_damp.Ko2.
            Note that for linear tuning, pi_pitch_reg3.K2 and aero_damp.Ko2
            are set to zero.
    """

    def parse_line(line, controller):
        split1 = line.split('=')
        var1 = split1[0].strip()
        try:
            val1 = float(split1[1].split('[')[0])
            contr_tuning[controller + '.' + var1] = val1
            if len(split1) > 2:
                var2 = split1[1].split(',')[1].strip()
                val2 = float(split1[2].split('[')[0])
                contr_tuning[controller + '.' + var2] = val2
        except IndexError:
            pass

    contr_tuning = {}

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                controller = 'pi_gen_reg1'
            elif i == 2:
                controller = 'pi_gen_reg2'
            elif i == 6:
                controller = 'pi_pitch_reg3'
            elif i == 10:
                controller = 'aero_damp'
            elif i < 13:
                parse_line(line, controller)

    # set some parameters to zero for the linear case
    try:
        contr_tuning['pi_pitch_reg3.K2']
    except KeyError:
        contr_tuning['pi_pitch_reg3.K2'] = 0.0
    try:
        contr_tuning['aero_damp.Ko2']
    except KeyError:
        contr_tuning['aero_damp.Ko2'] = 0.0

    return contr_tuning
