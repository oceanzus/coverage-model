#!/usr/bin/env python

"""
@package coverage_model.coverage
@file coverage_model/coverage.py
@author Christopher Mueller
@brief The core classes comprising the Coverage Model
"""

# http://collabedit.com/kkacp


# Notes:
# cell_map[cell_handle]=(0d_handles,)
# 0d_map[0d_handle]=(cell_handles,)
# ** Done via an association table of 2 arrays - indices are the alignment
#
# parameter values implicitly aligned to 0d handle array
#
# shape applied to 0d handle array
#
# cell is 'top level dimension'
#
# parameters can be implicitly aligned to cell array

# TODO: All implementation is 'pre-alpha' - intended primarily to flesh out the API (though some may stick around)

#CBM:TODO: Add type checking throughout all classes as determined appropriate, a la:
#@property
#def spatial_domain(self):
#    return self.__spatial_domain
#
#@spatial_domain.setter
#def spatial_domain(self, value):
#    if isinstance(value, AbstractDomain):
#        self.__spatial_domain = value

from pyon.public import log
from pyon.util.containers import DotDict

from coverage_model.basic_types import AbstractIdentifiable, AbstractBase, AxisTypeEnum, MutabilityEnum
from coverage_model.parameter import Parameter, ParameterDictionary
import numpy as np
import pickle

#=========================
# Coverage Objects
#=========================

class AbstractCoverage(AbstractIdentifiable):
    """
    Core data model, persistence, etc
    TemporalTopology
    SpatialTopology
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


    @classmethod
    def save(cls, cov_obj, file_path, use_ascii=False):
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))

        with open(file_path, 'w') as f:
            pickle.dump(cov_obj, f, 0 if use_ascii else 2)

        log.info('Saved coverage_model to \'%s\'', file_path)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as f:
            obj = pickle.load(f)

        if not isinstance(obj, AbstractCoverage):
            raise StandardError('loaded object must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(obj)))

        log.info('Loaded coverage_model from \'%s\'', file_path)
        return obj

    @classmethod
    def copy(cls, cov_obj, *args):
        raise NotImplementedError('Coverages cannot yet be copied. You can load multiple \'independent\' copies of the same coverage, but be sure to save them to different names.')


        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))

        # NTK:
        # Args need to have 1-n (ParameterContext, DomainOfApplication, DomainOfApplication,) tuples
        # Need to pull the parameter_dictionary, spatial_domain and temporal_domain from cov_obj (TODO: copies!!)
        # DOA's and PC's used to copy data - TODO: Need way of reshaping PC's?
        ccov = SimplexCoverage(name='', range_dictionary=None, spatial_domain=None, temporal_domain=None)

        return ccov


class ViewCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1 AbstractCoverage and applies a Filter
    """
    def __init__(self):
        AbstractCoverage.__init__(self)

        self.reference_coverage = ''
        self.structure_filter = StructureFilter()
        self.parameter_filter = ParameterFilter()

class ComplexCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1-n coverages
    """
    def __init__(self):
        AbstractCoverage.__init__(self)
        self.coverages = []

class SimplexCoverage(AbstractCoverage):
    """
    A concrete implementation of AbstractCoverage consisting of 2 domains (temporal and spatial)
    and a collection of parameters associated with one or both of the domains.  Each parameter is defined by a
    ParameterContext object (provided via the ParameterDictionary) and has content represented by a concrete implementation
    of the AbstractParameterValue class.

    """
    def __init__(self, name, parameter_dictionary, spatial_domain, temporal_domain=None):
        """
        Constructor for SimplexCoverage

        @param name    The name of the coverage
        @param parameter_dictionary    a ParameterDictionary object expected to contain one or more valid ParameterContext objects
        @param spatial_domain  a concrete instance of AbstractDomain for the spatial domain component
        @param temporal_domain a concrete instance of AbstractDomain for the temporal domain component
        """
        AbstractCoverage.__init__(self)

        self.name = name
        self.parameter_dictionary = parameter_dictionary
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain or GridDomain(GridShape('temporal',[0]), CRS.standard_temporal(), MutabilityEnum.EXTENSIBLE)
        self.range_dictionary = DotDict()
        self.range_value = DotDict()
        self._pcmap = {}
        self._temporal_param_name = None

        if isinstance(self.parameter_dictionary, ParameterDictionary):
            for p in self.parameter_dictionary:
                self._append_parameter(self.parameter_dictionary.get_context(p))

    def append_parameter(self, parameter_context):
        """
        Append a ParameterContext to the coverage

        @deprecated use a ParameterDictionary during construction of the coverage
        """
        log.warn('SimplexCoverage.append_parameter() is deprecated and will be removed shortly: use a ParameterDictionary during construction of the coverage')
        self._append_parameter(parameter_context)

    def _append_parameter(self, parameter_context):
        """
        Appends a ParameterContext object to the internal set for this coverage.

        The supplied ParameterContext is added to self.range_dictionary.  An AbstractParameterValue of the type
        indicated by ParameterContext.param_type is added to self.range_value.  If the ParameterContext indicates that
        the parameter is a coordinate parameter, it is associated with the indicated axis of the appropriate CRS.

        @param parameter_context    The ParameterContext to append to the coverage
        @throws StandardError   If the ParameterContext.axis indicates that it is temporal and a temporal parameter
        already exists in the coverage
        """
        pname = parameter_context.name

        # Determine the correct array shape (default is the shape of the spatial_domain)
        # If there is only one extent in the spatial domain and it's size is 0, collapse to time only
        # CBM TODO: This determination must be made based on the 'variability' of the parameter (temporal, spatial, both), not by assumption
        if len(self.spatial_domain.shape.extents) == 1 and self.spatial_domain.shape.extents[0] == 0:
            shp = self.temporal_domain.shape.extents
        else:
            shp = self.temporal_domain.shape.extents + self.spatial_domain.shape.extents

        # Assign the pname to the CRS (if applicable) and select the appropriate domain (default is the spatial_domain)
        dom = self.spatial_domain
        if not parameter_context.reference_frame is None and AxisTypeEnum.is_member(parameter_context.reference_frame, AxisTypeEnum.TIME):
            if self._temporal_param_name is None:
                self._temporal_param_name = pname
            else:
                raise StandardError("temporal_parameter already defined.")
            dom = self.temporal_domain
            shp = self.temporal_domain.shape.extents
            dom.crs.axes[parameter_context.reference_frame] = parameter_context.name
        elif parameter_context.reference_frame in self.spatial_domain.crs.axes:
            dom.crs.axes[parameter_context.reference_frame] = parameter_context.name

        self._pcmap[pname] = (len(self._pcmap), parameter_context, dom)
        self.range_dictionary[pname] = parameter_context
        self.range_value[pname] = RangeMember(shp, parameter_context)

    def get_parameter(self, param_name):
        """
        Get a Parameter object by name

        The Parameter object contains the ParameterContext and AbstractParameterValue associated with the param_name

        @param param_name  The local name of the parameter to return
        @returns A Parameter object containing the context and value for the specified parameter
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if param_name in self.range_dictionary:
            p = Parameter(self.range_dictionary[param_name], self._pcmap[param_name][2], self.range_value[param_name])
            return p
        else:
            raise KeyError('Coverage does not contain parameter \'{0}\''.format(param_name))

    def list_parameters(self, coords_only=False, data_only=False):
        """
        List the names of the parameters contained in the coverage

        @param coords_only List only the coordinate parameters
        @param data_only   List only the data parameters (non-coordinate) - superseded by coords_only
        @returns A list of parameter names
        """
        if coords_only:
            lst=[x for x, v in self.range_dictionary.iteritems() if v.is_coordinate]
        elif data_only:
            lst=[x for x, v in self.range_dictionary.iteritems() if not v.is_coordinate]
        else:
            lst=[x for x in self.range_dictionary.iterkeys()]
        lst.sort()
        return lst

    def insert_timesteps(self, count, origin=None):
        """
        Insert count # of timesteps beginning at the origin

        The specified # of timesteps are inserted into the temporal value array at the indicated origin.  This also
        expands the temporal dimension of the AbstractParameterValue for each parameters

        @param count    The number of timesteps to insert
        @param origin   The starting location, from which to begin the insertion
        """
        if not origin is None:
            raise SystemError('Only append is currently supported')

        # Expand the shape of the temporal_dimension
        shp = self.temporal_domain.shape
        shp.extents[0] += count

        # Expand the temporal dimension of each of the parameters that are temporal TODO: Indicate which are temporal!
        for n in self._pcmap:
            arr = self.range_value[n].content
            pc = self.range_dictionary[n]
            narr = np.empty((count,) + arr.shape[1:], dtype=pc.param_type.value_encoding)
            narr.fill(pc.fill_value)
            arr = np.append(arr, narr, 0)
            self.range_value[n].content = arr

    def set_time_values(self, value, tdoa):
        """
        Convenience method for setting time values

        @param value    The value to set
        @param tdoa The temporal DomainOfApplication; default to full Domain
        """
        return self.set_parameter_values(self._temporal_param_name, value, tdoa, None)

    def get_time_values(self, tdoa=None, return_value=None):
        """
        Convenience method for retrieving time values

        Delegates to get_parameter_values, supplying the temporal parameter name and sdoa == None
        @param tdoa The temporal DomainOfApplication; default to full Domain
        @param return_value If supplied, filled with response value
        """
        return self.get_parameter_values(self._temporal_param_name, tdoa, None, return_value)

    @property
    def num_timesteps(self):
        """
        The current number of timesteps
        """
        return self.temporal_domain.shape.extents[0]

    def set_parameter_values(self, param_name, value, tdoa=None, sdoa=None):
        """
        Assign value to the specified parameter

        Assigns the value to param_name within the coverage.  Temporal and spatial DomainOfApplication objects can be
        applied to constrain the assignment.  See DomainOfApplication for details

        @param param_name   The name of the parameter
        @param value    The value to set
        @param tdoa The temporal DomainOfApplication
        @param sdoa The spatial DomainOfApplication
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if not param_name in self.range_value:
            raise KeyError('Parameter \'{0}\' not found in coverage_model'.format(param_name))

        tdoa = _get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        sdoa = _get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)

        log.debug('Temporal doa: %s', tdoa.slices)
        log.debug('Spatial doa: %s', sdoa.slices)

        slice_ = []
        slice_.extend(tdoa.slices)
        slice_.extend(sdoa.slices)

        log.debug('Setting slice: %s', slice_)

        #TODO: Do we need some validation that slice_ is the same rank and/or shape as values?
        self.range_value[param_name][slice_] = value

    def get_parameter_values(self, param_name, tdoa=None, sdoa=None, return_value=None):
        """
        Retrieve the value for a parameter

        Returns the value from param_name.  Temporal and spatial DomainOfApplication objects can be used to
        constrain the response.  See DomainOfApplication for details.

        @param param_name   The name of the parameter
        @param tdoa The temporal DomainOfApplication
        @param sdoa The spatial DomainOfApplication
        @param return_value If supplied, filled with response value
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if not param_name in self.range_value:
            raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        return_value = return_value or np.zeros([0])

        tdoa = _get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        sdoa = _get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)

        log.debug('Temporal doa: %s', tdoa.slices)
        log.debug('Spatial doa: %s', sdoa.slices)

        slice_ = []
        slice_.extend(tdoa.slices)
        slice_.extend(sdoa.slices)

        log.debug('Getting slice: %s', slice_)

        return_value = self.range_value[param_name][slice_]
        return return_value

    def get_parameter_context(self, param_name):
        """
        Retrieve the ParameterContext object for the specified parameter

        @param param_name   The name of the parameter for which to retrieve context
        @returns A ParameterContext object
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if not param_name in self.range_dictionary:
            raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        return self.range_dictionary[param_name]

    @property
    def info(self):
        """
        Returns a detailed string representation of the coverage contents
        @returns    string of coverage contents
        """
        lst = []
        indent = ' '
        lst.append('ID: {0}'.format(self._id))
        lst.append('Name: {0}'.format(self.name))
        lst.append('Temporal Domain:\n{0}'.format(self.temporal_domain.__str__(indent*2)))
        lst.append('Spatial Domain:\n{0}'.format(self.spatial_domain.__str__(indent*2)))

        lst.append('Parameters:')
        for x in self.range_value:
            lst.append('{0}{1} {2}\n{3}'.format(indent*2,x,self.range_value[x].shape,self.range_dictionary[x].__str__(indent*4)))

        return '\n'.join(lst)

    def __str__(self):
        lst = []
        indent = ' '
        lst.append('ID: {0}'.format(self._id))
        lst.append('Name: {0}'.format(self.name))
        lst.append('TemporalDomain: Shape=>{0} Axes=>{1}'.format(self.temporal_domain.shape.extents, self.temporal_domain.crs.axes))
        lst.append('SpatialDomain: Shape=>{0} Axes=>{1}'.format(self.spatial_domain.shape.extents, self.spatial_domain.crs.axes))
        lst.append('Coordinate Parameters: {0}'.format(self.list_parameters(coords_only=True)))
        lst.append('Data Parameters: {0}'.format(self.list_parameters(coords_only=False, data_only=True)))

        return '\n'.join(lst)


#=========================
# Range Objects
#=========================

#TODO: Consider usage of Ellipsis in all this slicing stuff as well

class DomainOfApplication(object):
    # CBM: Document this!!
    def __init__(self, slices, topoDim=None):
        if slices is None:
            raise StandardError('\'slices\' cannot be None')
        self.topoDim = topoDim or 0

        if _is_valid_constraint(slices):
            if not np.iterable(slices):
                slices = [slices]

            self.slices = slices
        else:
            raise StandardError('\'slices\' must be either single, tuple, or list of slice or int objects')

    def __iter__(self):
        return self.slices.__iter__()

    def __len__(self):
        return len(self.slices)

def _is_valid_constraint(v):
    ret = False
    if isinstance(v, (slice, int)) or \
       (isinstance(v, (list,tuple)) and np.array([_is_valid_constraint(e) for e in v]).all()):
            ret = True

    return ret

def _get_valid_DomainOfApplication(v, valid_shape):
    """
    Takes the value to validate and a tuple representing the valid_shape
    """

    if v is None:
        if len(valid_shape) == 1:
            v = slice(None)
        else:
            v = [slice(None) for x in valid_shape]

    if not isinstance(v, DomainOfApplication):
        v = DomainOfApplication(v)

    return v

class RangeMember(object):
    # CBM TODO: Is this really AbstractParameterValue?? - I think so...content --> value
    """
    This is what would provide the "abstraction" between the in-memory model and the underlying persistence - all through __getitem__ and __setitem__
    Mapping between the "complete" domain and the storage strategy can happen here
    """

    def __init__(self, shape, pcontext):
        self._arr_obj = np.empty(shape, dtype=pcontext.param_type.value_encoding)
        self._arr_obj.fill(pcontext.fill_value)

    @property
    def shape(self):
        return self._arr_obj.shape

    @property
    def content(self):
        return self._arr_obj

    @content.setter
    def content(self, value):
        self._arr_obj=value

    # CBM: First swack - see this for more possible checks: http://code.google.com/p/netcdf4-python/source/browse/trunk/netCDF4_utils.py
    def __getitem__(self, slice_):
        if not _is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a tuple
        if not np.iterable(slice_):
            slice_ = (slice_,)
        elif not isinstance(slice_,tuple):
            slice_ = tuple(slice_)

        # Then make it's the correct shape TODO: Should reference the shape of the Domain object
        alen = len(self._arr_obj.shape)
        slen = len(slice_)
        if not slen == alen:
            if slen > alen:
                slice_ = slice_[:alen]
            else:
                for n in range(slen, alen):
                    slice_ += (slice(None,None,None),)

        return self._arr_obj[slice_]

    def __setitem__(self, slice_, value):
        if not _is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a tuple
        if not np.iterable(slice_):
            slice_ = (slice_,)
        elif not isinstance(slice_,tuple):
            slice_ = tuple(slice_)

        # Then make it's the correct rank TODO: Should reference the rank of the Domain object
        alen = len(self._arr_obj.shape)
        slen = len(slice_)
        if not slen == alen:
            if slen > alen:
                slice_ = slice_[:alen]
            else:
                for n in range(slen, alen):
                    slice_ += (slice(None,None,None),)

        self._arr_obj[slice_] = value

    def __str__(self):
        return '{0}'.format(self._arr_obj.shape)

class RangeDictionary(AbstractIdentifiable):
    """
    Currently known as Taxonomy with respect to the Granule & RecordDictionary
    May be synonymous with RangeType

    @deprecated DO NOT USE - Subsumed by ParameterDictionary
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


#=========================
# Abstract Parameter Value Objects
#=========================

class AbstractParameterValue(AbstractBase):
    """

    """
    def __init__(self):
        AbstractBase.__init__(self)

class AbstractSimplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)

class AbstractComplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)


#=========================
# CRS Objects
#=========================
class CRS(AbstractIdentifiable):
    """

    """
    def __init__(self, axes):
        AbstractIdentifiable.__init__(self)
        self.axes=Axes()
        for l in axes:
            # Add an axis member for the indicated axis
            try:
                self.axes[l] = None
            except KeyError as ke:
                raise SystemError('Unknown AxisType \'{0}\': {1}'.format(l,ke.message))

    @classmethod
    def standard_temporal(cls):
        return CRS([AxisTypeEnum.TIME])

    @classmethod
    def lat_lon_height(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    @classmethod
    def lat_lon(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    @classmethod
    def x_y_z(cls):
        return CRS([AxisTypeEnum.GEO_X, AxisTypeEnum.GEO_Y, AxisTypeEnum.GEO_Z])

    def _dump(self):
        ret = dict((k,v) for k,v in self.__dict__.iteritems())
        ret['cm_type'] = self.__class__.__name__
        return ret

    @classmethod
    def _load(cls, cdict):
        if isinstance(cdict, dict) and 'cm_type' in cdict and cdict['cm_type']:
            import inspect
            mod = inspect.getmodule(cls)
            ptcls=getattr(mod, cdict['cm_type'])

            args = inspect.getargspec(ptcls.__init__).args
            del args[0] # get rid of 'self'
            kwa={}
            for a in args:
                kwa[a] = cdict[a] if a in cdict else None

            ret = ptcls(**kwa)
            for k,v in cdict.iteritems():
                if not k in kwa.keys() and not k == 'cm_type':
                    setattr(ret,k,v)

            return ret
        else:
            raise TypeError('cdict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(cdict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Axes: {1}'.format(indent, self.axes))

        return '\n'.join(lst)

class Axes(dict):
    """
    Ensures that the indicated axis exists and that the string representation is used for the key
    """
    # CBM TODO: Think about if this complexity is really necessary - can it just print Enum names on __str__??
    def __getitem__(self, item):
        if item in AxisTypeEnum._str_map:
            item = AxisTypeEnum._str_map[item]
        elif item in AxisTypeEnum._value_map:
            pass
        else:
            raise KeyError('Invalid axis key, must be a member of AxisTypeEnum')

        return dict.__getitem__(self, item)

    def __setitem__(self, key, value):
        if key in AxisTypeEnum._str_map:
            key = AxisTypeEnum._str_map[key]
        elif key in AxisTypeEnum._value_map:
            pass
        else:
            raise KeyError('Invalid axis key, must be a member of AxisTypeEnum')

        dict.__setitem__(self, key, value)

    def __contains__(self, item):
        if item in AxisTypeEnum._str_map:
            item = AxisTypeEnum._str_map[item]

        return dict.__contains__(self, item)

#=========================
# Domain Objects
#=========================

class AbstractDomain(AbstractIdentifiable):
    """

    """
    def __init__(self, shape, crs, mutability=None):
        AbstractIdentifiable.__init__(self)
        self.shape = shape
        self.crs = crs
        self.mutability = mutability or MutabilityEnum.IMMUTABLE

#    def get_mutability(self):
#        return self.mutability

    def get_max_dimension(self):
        pass

    def get_num_elements(self, dim_index):
        pass

#    def get_shape(self):
#        return self.shape

    def insert_elements(self, dim_index, count, doa):
        pass

    def dump(self):
        ret={'cm_type':self.__class__.__name__}
        for k,v in self.__dict__.iteritems():
            if hasattr(v, '_dump'):
                ret[k]=v._dump()
            else:
                ret[k]=v

        return ret

    @classmethod
    def load(cls, ddict):
        """
        Create a concrete AbstractDomain from a dict representation

        @param cls  An AbstractDomain instance
        @param ddict   A dict containing information for a concrete AbstractDomain (requires a 'cm_type' key)
        """
        dd=ddict.copy()
        if isinstance(dd, dict) and 'cm_type' in dd:
            dd.pop('cm_type')
            #TODO: If additional required constructor parameters are added, make sure they're dealt with here
            crs = CRS._load(dd.pop('crs'))
            shp = AbstractShape._load(dd.pop('shape'))

            pc=AbstractDomain(shp,crs)
            for k, v in dd.iteritems():
                setattr(pc,k,v)

            return pc
        else:
            raise TypeError('ddict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(ddict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst=[]
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Shape:\n{1}'.format(indent, self.shape.__str__(indent*2)))
        lst.append('{0}CRS:\n{1}'.format(indent, self.crs.__str__(indent*2)))
        lst.append('{0}Mutability: {1}'.format(indent, MutabilityEnum._str_map[self.mutability]))

        return '\n'.join(lst)

class GridDomain(AbstractDomain):
    """

    """
    def __init__(self, shape, crs, mutability=None):
        AbstractDomain.__init__(self, shape, crs, mutability)

class TopologicalDomain(AbstractDomain):
    """

    """
    def __init__(self):
        AbstractDomain.__init__(self)


#=========================
# Shape Objects
#=========================

class AbstractShape(AbstractIdentifiable):
    """

    """
    def __init__(self, name, extents=None):
        AbstractIdentifiable.__init__(self)
        self.name = name
        self.extents = extents or [0]

    @property
    def rank(self):
        return len(self.extents)

#    def rank(self):
#        return len(self.extents)

    def _dump(self):
        ret = dict((k,v) for k,v in self.__dict__.iteritems())
        ret['cm_type'] = self.__class__.__name__
        return ret

    @classmethod
    def _load(cls, sdict):
        if isinstance(sdict, dict) and 'cm_type' in sdict and sdict['cm_type']:
            import inspect
            mod = inspect.getmodule(cls)
            ptcls=getattr(mod, sdict['cm_type'])

            args = inspect.getargspec(ptcls.__init__).args
            del args[0] # get rid of 'self'
            kwa={}
            for a in args:
                kwa[a] = sdict[a] if a in sdict else None

            ret = ptcls(**kwa)
            for k,v in sdict.iteritems():
                if not k in kwa.keys() and not k == 'cm_type':
                    setattr(ret,k,v)

            return ret
        else:
            raise TypeError('sdict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(sdict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}Extents: {1}'.format(indent, self.extents))

        return '\n'.join(lst)

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, extents=None):
        AbstractShape.__init__(self, name, extents)

    #CBM: Make extents type-safe


#=========================
# Filter Objects
#=========================

class AbstractFilter(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


class StructureFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)

class ParameterFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)



#=========================
# Possibly OBE ?
#=========================

#class Topology():
#    """
#    Sets of topological entities
#    Mappings between topological entities
#    """
#    pass
#
#class TopologicalEntity():
#    """
#
#    """
#    pass
#
#class ConstructionType():
#    """
#    Lattice or not (i.e. 'unstructured')
#    """
#    pass