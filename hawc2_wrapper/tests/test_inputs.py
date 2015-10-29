# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:21:10 2015

@author: dave
"""

import unittest

from hawc2_wrapper import hawc2_inputdict

class tests(unittest.TestCase):
    """
    """

    def setUp(self):
        self.fpath_linear = 'data/controller_input_linear.txt'
        self.fpath_quadratic = 'data/controller_input_quadratic.txt'

    def test_linear_file(self):

        res = hawc2_inputdict.read_controller_tuning_file(self.fpath_linear)

        self.assertEqual(res['pi_gen_reg1.K'], 0.108313E+07)

        self.assertEqual(res['pi_gen_reg2.I'], 0.307683E+08)
        self.assertEqual(res['pi_gen_reg2.Kp'], 0.135326E+08)
        self.assertEqual(res['pi_gen_reg2.Ki'], 0.303671E+07)

        self.assertEqual(res['pi_pitch_reg3.Kp'], 0.276246E+01)
        self.assertEqual(res['pi_pitch_reg3.Ki'], 0.132935E+01)
        self.assertEqual(res['pi_pitch_reg3.K1'], 5.79377)
        self.assertEqual(res['pi_pitch_reg3.K2'], 0.0)

        self.assertEqual(res['aero_damp.Kp2'], 0.269403E+00)
        self.assertEqual(res['aero_damp.Ko1'], -4.21472)
        self.assertEqual(res['aero_damp.Ko2'], 0.0)

    def test_quadratic_file(self):

        res = hawc2_inputdict.read_controller_tuning_file(self.fpath_quadratic)

        self.assertEqual(res['pi_gen_reg1.K'], 0.108313E+07)

        self.assertEqual(res['pi_gen_reg2.I'], 0.307683E+08)
        self.assertEqual(res['pi_gen_reg2.Kp'], 0.135326E+08)
        self.assertEqual(res['pi_gen_reg2.Ki'], 0.303671E+07)

        self.assertEqual(res['pi_pitch_reg3.Kp'], 0.249619E+01)
        self.assertEqual(res['pi_pitch_reg3.Ki'], 0.120122E+01)
        self.assertEqual(res['pi_pitch_reg3.K1'], 7.30949)
        self.assertEqual(res['pi_pitch_reg3.K2'], 1422.81187)

        self.assertEqual(res['aero_damp.Kp2'], 0.240394E-01)
        self.assertEqual(res['aero_damp.Ko1'], -1.69769)
        self.assertEqual(res['aero_damp.Ko2'], -15.02688)

if __name__ == '__main__':

    unittest.main()
