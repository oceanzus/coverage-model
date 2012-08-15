#!/usr/bin/env python

"""
@package coverage_model.test.examples
@file coverage_model/test/examples.py
@author Christopher Mueller
@brief Exemplar functions for creation, manipulation, and basic visualization of coverages
"""

from pyon.public import log
from netCDF4 import Dataset
from coverage_model.basic_types import *
from coverage_model.coverage import *
from coverage_model.parameter_types import *
from coverage_model.parameter import *
import numpy as np

def samplecov(save_coverage=True):
    # Instantiate a ParameterDictionary
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

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 1d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('sample coverage_model', pdict, sdom, tdom)

    # Insert some timesteps (automatically expands other arrays)
    scov.insert_timesteps(10)

    # Add data for each parameter
    scov.set_parameter_values('time', value=np.arange(10))
    scov.set_parameter_values('lat', value=45)
    scov.set_parameter_values('lon', value=-71)
    # make a random sample of 10 values between 23 and 26
    # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
    # --> To sample  multiply the output of random_sample by (b-a) and add a
    tvals=np.random.random_sample(10)*(26-23)+23
    scov.set_parameter_values('temp', value=tvals)
    scov.set_parameter_values('conductivity', value=np.random.random_sample(10)*(30-20)+20)

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/sample.cov')

    return scov

def roms2cov(save_coverage=True):
    ds = Dataset('test_data/roms.nc')
    
    var_names = ['ocean_time','lat_psi','lon_psi','s_rho','s_w','salt','temp','u','v',]
    
    #var_names = ['time','lat','lon','depth','water_u','water_v','salinity','water_temp',]
    
    pdict = ParameterDictionary()
    
    for v in var_names:
        var = ds.variables[v]
        
        pcontext = ParameterContext(v, param_type=QuantityType(value_encoding=ds.variables[v].dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the reference_frame for the coordinate parameters
        if v == 'ocean_time':
            pcontext.reference_frame = AxisTypeEnum.TIME
        elif v == 'lat_psi':
            pcontext.reference_frame = AxisTypeEnum.LAT
        elif v == 'lon_psi':
            pcontext.reference_frame = AxisTypeEnum.LON
        elif v == 's_rho':
            pcontext.reference_frame = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal'), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [129,36,81]), scrs, MutabilityEnum.IMMUTABLE) # 3d spatial topology (grid)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('sample grid coverage_model', pdict, sdom, tdom)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['ocean_time']
    scov.insert_timesteps(tvar.size)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
        # TODO: Sort out how to leave these sparse internally and only broadcast during read
        if v == 's_rho':
            z,_,_ = my_meshgrid(arr,np.zeros([36]),np.zeros([81]),indexing='ij',sparse=True)
            scov.range_value[v][:] = z
        elif v == 'lat_psi':
            _,y,_ = my_meshgrid(np.zeros([129]),arr,np.zeros([81]),indexing='ij',sparse=True)
            scov.range_value[v][:] = y
        elif v == 'lon_psi':
            _,_,x = my_meshgrid(np.zeros([129]),np.zeros([36]),arr,indexing='ij',sparse=True)
            scov.range_value[v][:] = x
        else:
            scov.range_value[v][:] = var[:]

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/roms.cov')

    return scov


def ncgrid2cov(save_coverage=True):
    # Open the netcdf dataset
    ds = Dataset('test_data/ncom.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','depth','water_u','water_v','salinity','water_temp',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionary
    for v in var_names:
        var = ds.variables[v]

        pcontext = ParameterContext(v, param_type=QuantityType(value_encoding=ds.variables[v].dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the reference_frame for the coordinate parameters
        if v == 'time':
            pcontext.reference_frame = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.reference_frame = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.reference_frame = AxisTypeEnum.LON
        elif v == 'depth':
            pcontext.reference_frame = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal'), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [34,57,89]), scrs, MutabilityEnum.IMMUTABLE) # 3d spatial topology (grid)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('sample grid coverage_model', pdict, sdom, tdom)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
        # TODO: Sort out how to leave these sparse internally and only broadcast during read
        if v == 'depth':
            z,_,_ = my_meshgrid(arr,np.zeros([57]),np.zeros([89]),indexing='ij',sparse=True)
            scov.range_value[v][:] = z
        elif v == 'lat':
            _,y,_ = my_meshgrid(np.zeros([34]),arr,np.zeros([89]),indexing='ij',sparse=True)
            scov.range_value[v][:] = y
        elif v == 'lon':
            _,_,x = my_meshgrid(np.zeros([34]),np.zeros([57]),arr,indexing='ij',sparse=True)
            scov.range_value[v][:] = x
        else:
            scov.range_value[v][:] = var[:]

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/ncom.cov')

    return scov

def ncstation2cov(save_coverage=True):
    # Open the netcdf dataset
    ds = Dataset('test_data/usgs.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','z','streamflow','water_temperature',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionaryl
    for v in var_names:
        var = ds.variables[v]

        pcontext = ParameterContext(v, param_type=QuantityType(var.dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the reference_frame for the coordinate parameters
        if v == 'time':
            pcontext.reference_frame = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.reference_frame = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.reference_frame = AxisTypeEnum.LON
        elif v == 'z':
            pcontext.reference_frame = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS.standard_temporal()
    scrs = CRS.lat_lon_height()

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 1d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('sample station coverage_model', pdict, sdom, tdom)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)

        # TODO: Sort out how to leave the coordinates sparse internally and only broadcast during read
        scov.range_value[v][:] = var[:]

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/usgs.cov')

    return scov

def direct_read():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.debug('<========= Query =========>')
    log.debug('\n>> All data for first timestep\n')
    slice_ = 0
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data\n')
    slice_ = (slice(None))
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data for every other timestep from 0 to 10\n')
    slice_ = (slice(0,10,2))
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data for first, sixth, eighth, thirteenth, and fifty-sixth timesteps\n')
    slice_ = [[(0,5,7,12,55)]]
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

def direct_write():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.debug('<========= Assignment =========>')

    slice_ = (slice(None))
    value = 22
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.range_value['streamflow'][slice_] = value
    log.debug(scov.range_value['streamflow'][slice_])

    slice_ = [[(1,5,7,)]]
    value = [10, 20, 30]
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.range_value['streamflow'][slice_] = value
    log.debug(scov.range_value['streamflow'][slice_])

def methodized_read():
    from coverage_model.test.examples import SimplexCoverage
    import numpy as np
    import os

    log.debug('============ Station ============')
    pth = 'test_data/usgs.cov'
    if not os.path.exists(pth):
        raise SystemError('Cannot proceed, \'{0}\' file must exist.  Run the \'ncstation2cov()\' function to generate the file.'.format(pth))

    cov=SimplexCoverage.load(pth)
    ra=np.zeros([0])
    log.debug('\n>> All data for first timestep\n')
    log.debug(cov.get_parameter_values('water_temperature',0,None,ra))
    log.debug('\n>> All data\n')
    log.debug(cov.get_parameter_values('water_temperature',None,None,None))
    log.debug('\n>> All data for second, fifth and sixth timesteps\n')
    log.debug(cov.get_parameter_values('water_temperature',[[1,4,5]],None,None))
    log.debug('\n>> First datapoint (in x) for every 5th timestep\n')
    log.debug(cov.get_parameter_values('water_temperature',slice(0,None,5),0,None))
    log.debug('\n>> First datapoint for first 10 timesteps, passing DOA objects\n')
    tdoa = DomainOfApplication(slice(0,10))
    sdoa = DomainOfApplication(0)
    log.debug(cov.get_parameter_values('water_temperature',tdoa,sdoa,None))

    log.debug('\n============ Grid ============')
    pth = 'test_data/ncom.cov'
    if not os.path.exists(pth):
        raise SystemError('Cannot proceed, \'{0}\' file must exist.  Run the \'ncstation2cov()\' function to generate the file.'.format(pth))

    cov=SimplexCoverage.load(pth)
    ra=np.zeros([0])
    log.debug('\n>> All data for first timestep\n')
    log.debug(cov.get_parameter_values('water_temp',0,None,ra))
    log.debug('\n>> All data\n')
    log.debug(cov.get_parameter_values('water_temp',None,None,None))
    log.debug('\n>> All data for first, fourth, and fifth timesteps\n')
    log.debug(cov.get_parameter_values('water_temp',[[0,3,4]],None,None))
    log.debug('\n>> Data from z=0, y=10, x=10 for every 2nd timestep\n')
    log.debug(cov.get_parameter_values('water_temp',slice(0,None,2),[0,10,10],None))
    log.debug('\n>> Data from z=0-10, y=10, x=10 for the first 2 timesteps, passing DOA objects\n')
    tdoa = DomainOfApplication(slice(0,2))
    sdoa = DomainOfApplication([slice(0,10),10,10])
    log.debug(cov.get_parameter_values('water_temp',tdoa,sdoa,None))

def methodized_write():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.debug('<========= Assignment =========>')

    slice_ = (slice(None))
    value = 22
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.set_parameter_values('streamflow',value=value,tdoa=slice_)

    slice_ = [[(1,5,7,)]]
    value = [10, 20, 30]
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.set_parameter_values('streamflow',value=value,tdoa=slice_)

#    raise NotImplementedError('Example not yet implemented')

def test_plot_1():
    from coverage_model.test.examples import SimplexCoverage
    import matplotlib.pyplot as plt

    cov=SimplexCoverage.load('test_data/usgs.cov')

    log.debug('Plot the \'water_temperature\' and \'streamflow\' for all times')
    wtemp = cov.get_parameter_values('water_temperature')
    wtemp_pc = cov.get_parameter_context('water_temperature')
    sflow = cov.get_parameter_values('streamflow')
    sflow_pc = cov.get_parameter_context('streamflow')
    times = cov.get_parameter_values('time')
    time_pc = cov.get_parameter_context('time')

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(times,wtemp)
    ax1.set_xlabel('{0} ({1})'.format(time_pc.name, time_pc.uom))
    ax1.set_ylabel('{0} ({1})'.format(wtemp_pc.name, wtemp_pc.uom))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(times,sflow)
    ax2.set_xlabel('{0} ({1})'.format(time_pc.name, time_pc.uom))
    ax2.set_ylabel('{0} ({1})'.format(sflow_pc.name, sflow_pc.uom))

    plt.show(0)

def test_plot_2():
    from coverage_model.test.examples import SimplexCoverage
    import matplotlib.pyplot as plt

    cov=SimplexCoverage.load('test_data/usgs.cov')

    log.debug('Plot the \'water_temperature\' and \'streamflow\' for all times')
    wtemp_param = cov.get_parameter('water_temperature')
    sflow_param = cov.get_parameter('streamflow')
    time_param = cov.get_parameter('time')

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(time_param.value[:],wtemp_param.value[:])
    ax1.set_xlabel('{0} ({1})'.format(time_param.name, time_param.context.uom))
    ax1.set_ylabel('{0} ({1})'.format(wtemp_param.name, wtemp_param.context.uom))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(time_param.value[:],sflow_param.value[:])
    ax2.set_xlabel('{0} ({1})'.format(time_param.name, time_param.context.uom))
    ax2.set_ylabel('{0} ({1})'.format(sflow_param.name, sflow_param.context.uom))

    plt.show(0)

# Based on scitools meshgrid
def my_meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from two or more coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
         If True a sparse grid is returned in order to conserve memory.
         Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in
        order to conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous arrays.
        Furthermore, more than one element of a broadcast array may refer to
        a single memory location.  If you need to write to the arrays, make
        copies first.

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing keyword
    argument.  Giving the string 'ij' returns a meshgrid with matrix indexing,
    while 'xy' returns a meshgrid with Cartesian indexing.  In the 2-D case
    with inputs of length M and N, the outputs are of shape (N, M) for 'xy'
    indexing and (M, N) for 'ij' indexing.  In the 3-D case with inputs of
    length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M,
    N, P) for 'ij' indexing.  The difference is illustrated by the following
    code snippet::

        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)

    copy_ = kwargs.get('copy', True)
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    if not indexing in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    if indexing == 'xy':
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


if __name__ == "__main__":
#    scov, _ = ncstation2cov()
#    log.debug(scov)
#
#    log.debug('\n=======\n')
#
#    gcov, _ = ncgrid2cov()
#    log.debug(gcov)

#    direct_read_write()
    methodized_read()

#    from coverage_model.coverage_model import AxisTypeEnum
#    axis = 'TIME'
#    log.debug(axis == AxisTypeEnum.TIME)

    pass

"""

from coverage_model.test.simple_cov import *
scov, ds = ncstation2cov()


"""