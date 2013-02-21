#!/usr/bin/env python

"""
@package
@file test_coverage_construction.py
@author James D. Case
@brief
"""

import random
import os
import shutil
import tempfile

from pyon.public import log

from nose.plugins.attrib import attr
from coverage_model import *
import numpy as np
from unittest import TestCase

TEST_DIR = os.path.join(tempfile.gettempdir(), 'cov_mdl_tests')

@attr('UNIT',group='cov')
class TestCoverageConstructionUnit(TestCase):

    @classmethod
    def setUpClass(cls):
        os.mkdir(TEST_DIR)

    @classmethod
    def tearDownClass(cls):
        # Removes temporary files
        # Comment this out if you need to inspect the HDF5 files.
        shutil.rmtree(TEST_DIR)

    def setUp(self):
        # Create temporary working directory for the persisted coverage tests
        self.working_dir = TEST_DIR

    def test_append_parameter(self):
        # Create a basic coverage
        scov = _make_samplecov(root_dir=self.working_dir)

        # Define a new parameter to add to the coverage
        parameter_name = 'turbidity'
        pc_in = ParameterContext(parameter_name, param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_in.uom = 'FTU'

        # Attempt to add the parameter
        scov.append_parameter(pc_in)

        # TODO: Check internal memory representation is accurate and fully formed
        self.assertTrue(parameter_name in scov._range_dictionary.keys())

        # TODO: Verify metadata and structure added to HDF5 persistence

        # Add data to the parameter
        nt = 30
        scov.set_parameter_values(parameter_name, value=np.arange(nt))

        # TODO: Verify the HDF5 brick file is created and fully formed after data is added.

        # TODO: Verify added parameter exists even after we load the SimplexCoverage

        scov.close()

def _make_parameter_dict():
    # Construct ParameterDictionary of various QuantityTypes
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.reference_frame = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lat_ctxt.reference_frame = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    pdict.add_context(lat_ctxt)

    lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lon_ctxt.reference_frame = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    pdict.add_context(lon_ctxt)

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
    temp_ctxt.uom = 'degree_Celsius'
    pdict.add_context(temp_ctxt)

    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    cond_ctxt.uom = 'unknown'
    pdict.add_context(cond_ctxt)

    return pdict

def _make_tcrs():
    # Construct temporal Coordinate Reference System object
    tcrs = CRS([AxisTypeEnum.TIME])
    return tcrs

def _make_scrs():
    # Construct spatial Coordinate Reference System object
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])
    return scrs

def _make_tdom(tcrs):
    # Construct temporal domain object
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    return tdom

def _make_sdom(scrs):
    # Create spatial domain object
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
    return sdom

def _make_samplecov(root_dir, mode=None, in_memory_storage=False, bricking_scheme=None, inline_data_writes=True, auto_flush_values=True, value_caching=True):
    # Instantiate a ParameterDictionary
    pdict = _make_parameter_dict()

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = _make_tcrs()
    scrs = _make_scrs()

    # Construct temporal and spatial Domain objects
    tdom = _make_tdom(tcrs)
    sdom = _make_sdom(scrs)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage(root_dir=root_dir, persistence_guid=create_guid(), name='sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, in_memory_storage=in_memory_storage, bricking_scheme=bricking_scheme, inline_data_writes=inline_data_writes, auto_flush_values=auto_flush_values, value_caching=value_caching)

    # Insert some timesteps (automatically expands other arrays)
    nt = 30
    scov.insert_timesteps(nt)

    # Add data for each parameter
    scov.set_parameter_values('time', value=np.arange(nt))
    scov.set_parameter_values('lat', value=45)
    scov.set_parameter_values('lon', value=-71)
    # make a random sample of 10 values between 23 and 26
    # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
    # --> To sample  multiply the output of random_sample by (b-a) and add a
    tvals=np.random.random_sample(nt)*(26-23)+23
    scov.set_parameter_values('temp', value=tvals)
    scov.set_parameter_values('conductivity', value=np.random.random_sample(nt)*(110-90)+90)

    return scov