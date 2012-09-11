#!/usr/bin/env python

"""
@package coverage_model.persistence
@file coverage_model/persistence.py
@author James Case
@brief The core classes comprising the Persistence Layer
"""

from pyon.public import log
from coverage_model.basic_types import *
from coverage_model.parameter_types import *
from coverage_model.parameter import ParameterDictionary
import sys, traceback
import tables
import numpy as np
import h5py
import os
import rtree
import itertools
import subprocess
import cStringIO

class PersistenceLayer():
    def __init__(self, root, guid, parameter_dictionary, tdom, sdom=None, *kwargs):
        """
        Constructor for PersistenceLayer
        """
        self.root = root
        self.guid = guid
        self.parameter_dictionary = parameter_dictionary
        self.tdom = tdom
        self.sdom = sdom

        brickTree = rtree.Rtree()
        self.brickTreeDict = {}
        self.brickCount = 0
        self.brickList = {}

        self.parameterDomainDict = {}

        self.param_dtype = np.empty((1,), dtype='f').dtype

        # TODO: Loop through parameter_dictionary
        if isinstance(self.parameter_dictionary, ParameterDictionary):
            for param in self.parameter_dictionary:
                pc = self.parameter_dictionary.get_context(param)

                if (self.sdom==None): # Stream coverage
                    # Total domain is always this for a coverage w/o a sdom
                    tD = ((0,),(0,))
                    # TODO: Calculate brick domain (bD)
                    bD = ((0,),(0,))
                    # TODO: Calculate chunk domain (cD)
                    cD = ((0,),(0,))
                    self.init_parameter(tD, bD, cD, pc.name, pc.dtype)
                else: # All other coverages
                    # TODO: Calculate total domain (tD)
                    tD = ((0,),(0,))
                    # TODO: Calculate brick domain (bD)
                    bD = ((0,),(0,))
                    # TODO: Calculate chunk domain (cD)
                    cD = ((0,),(0,))
                    self.init_parameter(tD, bD, cD, pc.name, pc.dtype)
        elif isinstance(self.parameter_dictionary, list):
            log.debug('Found a list of parameters, assuming all have the same total domain')
        elif isinstance(self.parameter_dictionary, dict):
            log.debug('Found a dictionary of parameters, assuming parameter name is key and has value of total domain')
            for pname,tD in self.parameter_dictionary:
                tD = list(self.tdom+self.sdom) #can increase
                bD,cD = self.calculate_brick_size(64) #remains same for each parameter
                self.parameterDomainDict[pname] = [tD,bD,cD]
                # Verify domain is Rtree friendly
                if len(bD) > 1:
                    p = rtree.index.Property()
                    p.dimension = len(bD)
                    brickTree = rtree.index.Index(properties=p)
                    self.brickTreeDict[pname] = [brickTree,self.tD]
                self.init_parameter(tD,bD,cD,pname,self.param_dtype)
        else:
            log.debug('No parameter_dictionary defined.  Running a test script...')
            if self.sdom==None:
                tD = list(self.tdom)
            else:
                tD = list(self.tdom+self.sdom) #can increase
            bD,cD = self.calculate_brick_size(64) #remains same for each parameter
            self.parameterDomainDict['Test Parameter'] = [tD,bD,cD]

            # Verify domain is Rtree friendly
            if len(bD) > 1:
                p = rtree.index.Property()
                p.dimension = len(bD)
                brickTree = rtree.index.Index(properties=p)
                self.brickTreeDict['Test Parameter'] = [brickTree,tD]
            self.init_parameter(tD,bD,cD,'Test Parameter',self.param_dtype)

    # Calculate brick domain size given a target file system brick size (Mbytes) and dtype
    def calculate_brick_size(self, target_fs_size):
        log.debug('Calculating the size of a brick...')

        # TODO: Hardcoded!!!!!!!!!!
        if self.sdom==None:
            bD = [10]
            cD = tuple([5])
        else:
            bD = [10]+list(self.sdom)
            cD = tuple([5]+list(self.sdom))

        return bD,cD
    # Generate empty bricks
    # Input: totalDomain, brickDomain, chunkDomain, parameterName, 0=do not write brick, 1=write brick
    def init_parameter(self, tD, bD, cD, parameterName, dataType):
        log.debug('Total Domain: {0}'.format(tD))
        log.debug('Brick Domain: {0}'.format(bD))
        log.debug('Chunk Domain: {0}'.format(cD))

        try:
            # Gather block list
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: {0}'.format(len(vertices)))

                # Write brick to HDF5 file
                # TODO: Loop over self.parameter_dictionary
                map(lambda origin: self.write_brick(origin,bD,cD,parameterName,dataType), vertices)

                log.debug('Created {0} Bricks'.format(self.brickCount))
                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain in empty...')
        except:
            log.error('Failed to Initialize Persistence Layer')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            log.error('{0}'.format(repr(traceback.format_exception(exc_type, exc_value, exc_traceback))))

            log.debug('Cleaning up bricks.')
            # TODO: Add brick cleanup routine

    # Write empty HDF5 brick to the filesystem
    # Input: Brick origin , brick dimensions (topological), chunk dimensions, coverage GUID, parameterName
    # TODO: ParameterContext (for HDF attributes)
    def write_brick(self,origin,bD,cD,parameterName,dataType):

        # Calculate the brick extents
        brickMax = []
        for idx,val in enumerate(origin):
            brickMax.append(bD[idx]+val)

        brickExtents = list(origin)+brickMax
        log.debug('Brick extents (rtree format): {0}'.format(brickExtents))

        # Make sure the brick doesn't already exist if we already have some bricks
        if len(self.brickList)>0:
            check = [(brickExtents==val) for guid,val in self.brickList.values()]
            if True in check:
                log.debug('Brick already exists!')
            else:
                self._write_brick(origin,bD,cD,parameterName,dataType)
        else:
            self._write_brick(origin,bD,cD,parameterName,dataType)


    def _write_brick(self,origin,bD,cD,parameterName,dataType):
        # Calculate the brick extents
        brickMax = []
        for idx,val in enumerate(origin):
            brickMax.append(bD[idx]+val)

        brickExtents = list(origin)+brickMax
        log.debug('Brick extents (rtree format): {0}'.format(brickExtents))

        self.brickCount = self.brickCount+1

        log.debug('Writing brick for parameter {0}'.format(parameterName))
        log.debug('Brick origin: {0}'.format(origin))

        rootPath = '{0}/{1}/{2}'.format(self.root,self.guid,parameterName)

        # Create the root path if it does not exist
        # TODO: Eliminate possible race condition
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        # Create a GUID for the brick
        brickGUID = create_guid()

        # Set HDF5 file and group
        sugarFileName = '{0}.hdf5'.format(brickGUID)
        sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
        sugarFile = h5py.File(sugarFilePath, 'w')

        sugarGroupPath = '/{0}/{1}'.format(self.guid,parameterName)
        sugarGroup = sugarFile.create_group(sugarGroupPath)

        # Create the HDF5 dataset that represents one brick
        sugarCubes = sugarGroup.create_dataset('{0}'.format(brickGUID), bD, dtype=dataType, chunks=cD)

        # Close the HDF5 file that represents one brick
        log.debug('Size Before Close: {0}'.format(os.path.getsize(sugarFilePath)))
        sugarFile.close()
        log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

        # Verify domain is Rtree friendly
        if len(bD) > 1:
            log.debug('Inserting into Rtree {0}:{1}:{2}'.format(self.brickCount,brickExtents,brickGUID))
            self.brickTreeDict[parameterName][0].insert(self.brickCount,brickExtents,obj=brickGUID)

        # Update the brick listing
        self.brickList[self.brickCount]=[brickGUID,brickExtents]

    # Expand the domain
    # TODO: Verify brick and chunk sizes are still valid????
    def expand_domain(self, parameterName, newDomain):
        log.debug('Placeholder for expanding a domain in any dimension')

        tD = self.parameterDomainDict[parameterName][0]
        bD = self.parameterDomainDict[parameterName][1]
        cD = self.parameterDomainDict[parameterName][2]

        deltaDomain = [(x - y) for x, y in zip(newDomain, tD)]
        log.debug('delta domain: {0}'.format(deltaDomain))

        tD = [(x + y) for x, y in zip(tD, deltaDomain)]
        self.parameterDomainDict[parameterName][0] = tD

        self.init_parameter(tD, bD, cD, parameterName, self.param_dtype)

    # Retrieve all or subset of data from HDF5 bricks
    def get_values(self, parameterName, minExtents, maxExtents):
        log.debug('Getting value(s) from brick(s)...')

        # Find bricks for given extents
        brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)
        log.debug('Found bricks that may contain data: {0}'.format(brickSearchList))

        # Figure out slices for each brick

        # Get the data (if it exists, jagged?) for each sliced brick

        # Combine the data into one numpy array

        # Pass back to coverage layer

    # Write all or subset of Coverage's data to HDF5 brick(s)
    def set_values(self, parameterName, payload, minExtents, maxExtents):
        log.debug('Setting value(s) of payload to brick(s)...')

        # TODO:  Make sure the content's domain has a brick available, otherwise make more bricks (expand)
        brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)

        if len(brickSearchList)==0:
            log.debug('No existing bricks found, creating now...')
            self.expand_domain(maxExtents)
            brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)

        if len(brickSearchList) > 1:
            log.debug('Splitting data across multiple bricks: {0}'.format(brickSearchList))
            # TODO: have to split data across multiple bricks
        else:
            log.debug('Writing all data to one brick: {0}'.format(brickSearchList))
            # TODO: all data goes in one brick
            # TODO: open brick and place the data in the dataset

    # List bricks for a parameter based on domain range
    def list_bricks(self, parameterName, start, end):
        log.debug('Placeholder for listing bricks based on a domain range...')
        hits = list(self.brickTreeDict[parameterName][0].intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]
