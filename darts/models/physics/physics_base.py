import hashlib
import os
import pickle
import atexit

from darts.engines import *

"""
    Base physics class with common utility functions

"""


class PhysicsBase:
    def __init__(self, cache=True):
        self.cache = cache
        # list of created interpolators
        # is used on destruction to save cache data
        if self.cache:
            self.created_itors = []
            atexit.register(self.write_cache)

    """
        Create interpolator object according to specified parameters

        Parameters
        ----------
        evaluator : an operator_set_evaluator_iface object
            State operators to be interpolated. Evaluator object is used to generate supporting points
        n_dims : integer
            The number of dimensions for interpolation (parameter space dimensionality)
        n_ops : integer
            The number of operators to be interpolated. Should be consistent with evaluator.
        axes_n_points: an index_vector, pybind-type vector of integers
            The number of supporting points for each axis.
        axes_min : a value_vector, pybind-type vector of floats
            The minimum value for each axis.
        axes_max : a value_vector, pybind-type vector of floats
            The maximum value for each axis.
        type : string
            interpolator type:
            'multilinear' (default) - piecewise multilinear generalization of piecewise bilinear interpolation on
                                      rectangles
            'linear' - a piecewise linear generalization of piecewise linear interpolation on triangles
        type : string
            interpolator mode:
            'adaptive' (default) - only supporting points required to perform interpolation are evaluated on-the-fly
            'static' - all supporting points are evaluated during itor object construction
        platform : string
            platform used for interpolation calculations :
            'cpu' (default) - interpolation happens on CPU
            'gpu' - interpolation happens on GPU
        precision : string
            precision used in interpolation calculations:
            'd' (default) - supporting points are stored and interpolation is performed using double precision
            's' - supporting points are stored and interpolation is performed using single precision
    """

    def create_interpolator(self, evaluator: operator_set_evaluator_iface, n_dims: int, n_ops: int,
                            axes_n_points: index_vector, axes_min: value_vector, axes_max: value_vector,
                            algorithm: str = 'multilinear', mode: str = 'adaptive',
                            platform: str = 'cpu', precision: str = 'd'):
        # verify then inputs are valid
        assert len(axes_n_points) == n_dims
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in axes_n_points:
            assert n_p > 1

        # calculate object name using 32 bit index type (i)
        itor_name = "%s_%s_%s_interpolator_i_%s_%d_%d" % (algorithm,
                                                          mode,
                                                          platform,
                                                          precision,
                                                          n_dims,
                                                          n_ops)
        itor = None
        general = False
        cache_loaded = 0
        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
        except (ValueError, NameError):
            # 32-bit index type did not succeed: either total amount of points is out of range or has not been compiled
            # try 64 bit now raising exception this time if goes wrong:
            itor_name = itor_name.replace('interpolator_i', 'interpolator_l')
            try:
                itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
            except (ValueError, NameError):
                # if 64-bit index also failed, probably the combination of required n_ops and n_dims
                # was not instantiated/exposed. In this case substitute general implementation of interpolator
                itor = eval("multilinear_adaptive_cpu_interpolator_general")(evaluator, axes_n_points, axes_min, axes_max,
                                                                             n_dims, n_ops)
                general = True

        if self.cache:
            # create unique signature for interpolator
            itor_cache_signature = "%s_%s_%s_%d_%d" % (type(evaluator).__name__, mode, precision, n_dims, n_ops)
            # geenral itor has a different point_data format
            if general:
                itor_cache_signature += "_general_"
            for dim in range(n_dims):
                itor_cache_signature += "_%d_%e_%e" % (axes_n_points[dim], axes_min[dim], axes_max[dim])
            # compute signature hash to uniquely identify itor parameters and load correct cache
            itor_cache_signature_hash = str(hashlib.md5(itor_cache_signature.encode()).hexdigest())
            itor_cache_filename = 'obl_point_data_' + itor_cache_signature_hash + '.pkl'

            # if cache file exists, read it
            if os.path.exists(itor_cache_filename):
                with open(itor_cache_filename, "rb") as fp:
                    print("Reading cached point data for ", type(itor).__name__)
                    itor.point_data = pickle.load(fp)
                    cache_loaded = 1
            if mode == 'adaptive':
                # for adaptive itors, delay obl data save moment, because
                # during simulations new points will be evaluated.
                # on model destruction (or interpreter exit), itor point data will be written to disk
                self.created_itors.append((itor, itor_cache_filename))

        itor.init()
        # for static itors, save the cache immediately after init, if it has not been already loaded
        # otherwise, there is no point to save the same data over and over
        if self.cache and mode == 'static' and not cache_loaded:
            with open(itor_cache_filename, "wb") as fp:
                print("Writing point data for ", type(itor).__name__)
                pickle.dump(itor.point_data, fp, protocol=4)
        return itor

    """
            Create timers for interpolators.

            Parameters
            ----------
            itor : an operator_set_gradient_evaluator_iface object
                The object which performes evaluation of operator gradient (interpolators currently, AD-based in future) 
            timer_name: string
                Timer name to be used for the given interpolator
        """

    def create_itor_timers(self, itor, timer_name: str):

        try:
            # in case this is a subsequent call, create only timer node for the given timer
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()
        except:
            # in case this is first call, create first only timer nodes for jacobian assembly and interpolation
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()

        # assign created timer to interpolator
        itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name])

    def __del__(self):
        # first write cache
        if self.cache:
            self.write_cache()
        # Now destroy all objects in physics
        for name in list(vars(self).keys()):
            delattr(self, name)

    def write_cache(self):
        # this function can be called two ways
        #   1. Destructor (__del__) method
        #   2. Via atexit function, before interpreter exits
        # In either case it should only be invoked by the earliest call (which can be 1 or 2 depending on situation)
        # Switch cache off to prevent the second call
        self.cache = False
        for itor, filename in self.created_itors:
            with open(filename, "wb") as fp:
                print("Writing point data for ", type(itor).__name__)
                pickle.dump(itor.point_data, fp, protocol=4)
