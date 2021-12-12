#!/usr/bin/env python
""" Tests of Larch Scripts  """
import unittest
import time
import ast
import numpy as np
import os
from sys import version_info

from utils import TestCase
from larch import Interpreter
class TestScripts(TestCase):
    '''tests'''

    def test_basic_interp(self):
        self.runscript('interp.lar', dirname='../examples/basic/')
        assert(len(self.session.get_errors()) == 0)
        self.isNear("y0[1]", 0.48578, places=3)
        self.isNear("y1[1]", 0.81310, places=3)
        self.isNear("y2[1]", 0.41532, places=3)
        


if __name__ == '__main__':  # pragma: no cover
    for suite in (TestScripts,):
        suite = unittest.TestLoader().loadTestsFromTestCase(suite)
        unittest.TextTestRunner(verbosity=13).run(suite)
