#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_coverage.py
@author James Case
@brief Test cases for the coverage_model module
"""
import random
import os
import shutil
import tempfile

from pyon.public import log

import numpy as np
from coverage_model import *
from nose.plugins.attrib import attr
from unittest import TestCase

@attr('UNIT', group='cov')
class TestCoverageModelUnit(TestCase):

    def setUp(self):
        # Create temporary working directory for the persisted coverage tests
        self.working_dir = tempfile.mkdtemp()

    def tearDown(self):
        pass

    def test_crs_epsg_valid(self):
        """
        Check that for a valid EPSG code the proper OGC-WKT is returned and both
         ogcwkt and epsg_code are set properly
        """
        scrs = CRS(axis_types=[AxisTypeEnum.LON, AxisTypeEnum.LAT], epsg_code=4326)
        expected_ogcwkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        self.assertEqual(scrs.ogcwkt, expected_ogcwkt)
        self.assertEqual(scrs.epsg_code, 4326)

    def test_crs_epsg_invalid(self):
        """
        Check that for an invalid EPSG code the ogc_wkt variable is set to None
        The epsg_code will always be set to the user-defined input regardless of validity
        """
        scrs = CRS(axis_types=[AxisTypeEnum.LON, AxisTypeEnum.LAT], epsg_code=43269075646)
        self.assertEqual(scrs.ogcwkt, None)
        self.assertEqual(scrs.epsg_code, 43269075646)