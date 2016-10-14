"""
Definition of SDF Class
Author: Sahaana Suri & Jeff Mahler

**Currently assumes clean input**
"""
from abc import ABCMeta, abstractmethod
import logging
import numpy as np

from PIL import Image
import scipy.io
import scipy.ndimage

import sdf_file as sf
import tfx
import time

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import random
import string

from sys import version_info
if version_info[0] != 3:
    range = xrange

MAX_CHAR = 255


class Sdf:
    __metaclass__ = ABCMeta

    ##################################################################
    # General SDF Properties
    ##################################################################
    @property
    def dimensions(self):
        """SDF dimension information.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            The ndarray that contains the dimensions of the sdf.
        """
        return self.dims_

    @property
    def origin(self):
        """The location of the origin in the SDF grid.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 2- or 3-ndarray that contains the location of
            the origin of the mesh grid in real space.
        """
        return self.origin_

    @property
    def resolution(self):
        """The grid resolution (how wide each grid cell is).

        Returns
        -------
        float
            The width of each grid cell.
        """
        return self.resolution_

    @property
    def center(self):
        """Center of grid.

        This basically transforms the world frame to grid center.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return self.center_

    @property
    def gradients(self):
        """Gradients of the SDF.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray` of float
            A list of ndarrays of the same dimension as the SDF. The arrays
            are in axis order and specify the gradients for that axis
            at each point.
        """
        return self.gradients_

    @property
    def data(self):
        """The SDF data.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 2- or 3-dimensional ndarray that holds the grid of signed
            distances.
        """
        return self.data_

    @property
    def tf(self):
        """Returns the transform of the SDF with respect to the world frame.

        Returns:
            TODO similarity transform: sdf tf
        """
        return self.tf_

    @tf.setter
    def tf(self, tf):
        self.tf_.tf = tf

    @property
    def pose(self):
        """Returns the pose of the SDF with respect to the world frame.

        Returns
        -------
            TODO tfx pose: sdf pose
        """
        return self.tf_.pose

    @pose.setter
    def pose(self, pose):
        self.tf_.pose = pose

    @property
    def scale(self):
        """Returns the scale of the SDF with respect to the world frame.

        Returns
        -------
        float
            The scale of the SDF.
        """
        return self.tf_.scale

    @scale.setter
    def scale(self, scale):
        self.tf_.scale = scale

    ##################################################################
    # General SDF Abstract Methods
    ##################################################################
    @abstractmethod
    def transform(self, tf):
        """Returns a new SDF transformed by similarity tf |tf| TODO
        """
        pass

    @abstractmethod
    def transform_pt_obj_to_grid(self, x_world, direction = False):
        """Transforms points from world frame to grid frame
        """
        pass

    @abstractmethod
    def transform_pt_grid_to_obj(self, x_grid, direction = False):
        """Transforms points from grid frame to world frame
        """
        pass

    @abstractmethod
    def surface_points(self):
        """Returns the points on the surface.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of float
            The points on the surface and the signed distances at those points.
        """
        pass

    @abstractmethod
    def __getitem__(self, coords):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).
        """
        pass

    ##################################################################
    # Universal SDF Methods
    ##################################################################
    def transform_to_world(self):
        """Returns an sdf object with center in the world frame of reference.
        """
        return self.transform(self.pose_, scale=self.scale_)

    def center_world(self):
        """Center of grid (basically transforms world frame to grid center)
        """
        return self.transform_pt_grid_to_obj(self.center_)

    def on_surface(self, coords):
        """Determines whether or not a point is on the object surface.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        :obj:`tuple` of bool, float
            Is the point on the object's surface, and what
            is the signed distance at that point?
        """
        sdf_val = self[coords]
        if np.abs(sdf_val) < self.surface_thresh_:
            return True, sdf_val
        return False, sdf_val

    def is_out_of_bounds(self, coords):
        """Returns True if coords is an out of bounds access.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        bool
            Are the coordinates in coords out of bounds?
        """
        return np.array(coords < 0).any() or np.array(coords >= self.dims_).any()

    def _compute_gradients(self):
        """Computes the gradients of the SDF.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray` of float
            A list of ndarrays of the same dimension as the SDF. The arrays
            are in axis order and specify the gradients for that axis
            at each point.
        """
        self.gradients_ = np.gradient(self.data_)

class Sdf3D(Sdf):
    # static indexing vars
    num_interpolants = 8
    min_coords_x = [0, 2, 3, 5]
    max_coords_x = [1, 4, 6, 7]
    min_coords_y = [0, 1, 3, 6]
    max_coords_y = [2, 4, 5, 7]
    min_coords_z = [0, 1, 2, 4]
    max_coords_z = [3, 5, 6, 7]

    def __init__(self, sdf_data, origin, resolution, tf = stf.SimilarityTransform3D(tfx.identity_tf(), scale = 1.0), frame = None, use_abs = True):
        self.data_ = sdf_data
        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape

        # set up surface params
        self.surface_thresh_ = self.resolution_ * np.sqrt(2) / 2 # resolution is max dist from surface when surf is orthogonal to diagonal grid cells
        spts, _ = self.surface_points()
        self.center_ = 0.5 * (np.min(spts, axis=0) + np.max(spts, axis=0))
        self.points_buf_ = np.zeros([Sdf3D.num_interpolants, 3], dtype=np.int)
        self.coords_buf_ = np.zeros([3,])

        # set up tf
        self.tf_ = tf

        # tranform sdf basis to grid (X and Z axes are flipped!)
        R_sdf_mesh = np.eye(3)
        self.tf_grid_sdf_ = stf.SimilarityTransform3D(tfx.canonical.CanonicalTransform(R_sdf_mesh, -R_sdf_mesh.T.dot(self.center_)), 1.0 / self.resolution_)
        self.tf_sdf_grid_ = self.tf_grid_sdf_.inverse()

        # optionally use only the absolute values (useful for non-closed meshes in 3D)
        if use_abs:
            self.data_ = np.abs(self.data_)

        self._compute_flat_indices()
        self._compute_gradients()

        self.feature_vector_ = None #Kmeans feature representation

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind, z_ind] = np.indices(self.dims_)
        self.pts_ = np.c_[x_ind.flatten().T, np.c_[y_ind.flatten().T, z_ind.flatten().T]]

    def _signed_distance(self, coords):
        """Returns the signed distance at the given coordinates, interpolating
        if necessary.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        pass
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional') 
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        self.coords_buf_[0] = max(0, min(coords[0], self.dims_[0] - 1))
        self.coords_buf_[1] = max(0, min(coords[1], self.dims_[1] - 1))
        self.coords_buf_[2] = max(0, min(coords[2], self.dims_[2] - 1))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int and type(coords[2]) is int:
            self.coords_buf_ = self.coords_buf_.astype(np.int)
            return self.data_[self.coords_buf_[0], self.coords_buf_[1], self.coords_buf_[2]]

        # otherwise interpolate
        min_coords = np.floor(self.coords_buf_)
        max_coords = min_coords + 1 # assumed to be on grid
        self.points_buf_[Sdf3D.min_coords_x, 0] = min_coords[0]
        self.points_buf_[Sdf3D.max_coords_x, 0] = max_coords[0]
        self.points_buf_[Sdf3D.min_coords_y, 1] = min_coords[1]
        self.points_buf_[Sdf3D.max_coords_y, 1] = max_coords[1]
        self.points_buf_[Sdf3D.min_coords_z, 2] = min_coords[2]
        self.points_buf_[Sdf3D.max_coords_z, 2] = max_coords[2]

        # bilinearly interpolate points
        sd = 0.0
        for i in range(Sdf3D.num_interpolants):
            p = self.points_buf_[i,:]
            if self.is_out_of_bounds(p):
                v = 0.0
            else:
                v = self.data_[p[0], p[1], p[2]]
            w = np.prod(-np.abs(p - self.coords_buf_) + 1)
            sd = sd + w * v

        return sd

    def __getitem__(self, coords):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The signed distance at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        return self._signed_distance(coords)

    def gradient(self, coords):
        """Returns the SDF gradient at the given coordinates, interpolating if necessary

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float
            The gradient at the given coords (interpolated).

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')

        # log warning if out of bounds access
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        self.coords_buf_[0] = max(0, min(coords[0], self.dims_[0] - 1))
        self.coords_buf_[1] = max(0, min(coords[1], self.dims_[1] - 1))
        self.coords_buf_[2] = max(0, min(coords[2], self.dims_[2] - 1))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int and type(coords[2]) is int:
            self.coords_buf_ = self.coords_buf_.astype(np.int)
            return self.data_[self.coords_buf_[0], self.coords_buf_[1], self.coords_buf_[2]]

        # otherwise interpolate
        min_coords = np.floor(self.coords_buf_)
        max_coords = min_coords + 1
        self.points_buf_[Sdf3D.min_coords_x, 0] = min_coords[0]
        self.points_buf_[Sdf3D.max_coords_x, 0] = min_coords[0]
        self.points_buf_[Sdf3D.min_coords_y, 1] = min_coords[1]
        self.points_buf_[Sdf3D.max_coords_y, 1] = max_coords[1]
        self.points_buf_[Sdf3D.min_coords_z, 2] = min_coords[2]
        self.points_buf_[Sdf3D.max_coords_z, 2] = max_coords[2]

        # bilinear interpolation
        g = np.zeros(3)
        gp = np.zeros(3)
        w_sum = 0.0
        for i in range(Sdf3D.num_interpolants):
            p = self.points_buf_[i,:]
            if self.is_out_of_bounds(p):
                gp[0] = 0.0
                gp[1] = 0.0
                gp[2] = 0.0
            else:
                gp[0] = self.gradients_[0][p[0], p[1], p[2]]
                gp[1] = self.gradients_[1][p[0], p[1], p[2]]
                gp[2] = self.gradients_[2][p[0], p[1], p[2]]

            w = np.prod(-np.abs(p - self.coords_buf_) + 1)
            g = g + w * gp

        return g

    # TODO THIS IS NOT USED. JEFF > COMMENT
    def curvature(self, coords, delta=0.001):
        """Returns an approximation to the local SDF curvature (Hessian) at the
        given coordinate in GRID BASIS.
        Returns:
            float: the approximate hessian (interpolated)
        """
        # perturb local coords
        coords_x_up   = coords + np.array([delta, 0, 0])
        coords_x_down = coords + np.array([-delta, 0, 0])
        coords_y_up   = coords + np.array([0, delta, 0])
        coords_y_down = coords + np.array([0, -delta, 0])
        coords_z_up   = coords + np.array([0, 0, delta])
        coords_z_down = coords + np.array([0, 0, -delta])

        # get gradient
        grad_x_up = self.gradient(coords_x_up)
        grad_x_down = self.gradient(coords_x_down)
        grad_y_up = self.gradient(coords_y_up)
        grad_y_down = self.gradient(coords_y_down)
        grad_z_up = self.gradient(coords_z_up)
        grad_z_down = self.gradient(coords_z_down)

        # finite differences
        curvature_x = (grad_x_up - grad_x_down) / (4 * delta)
        curvature_y = (grad_y_up - grad_y_down) / (4 * delta)
        curvature_z = (grad_z_up - grad_z_down) / (4 * delta)
        # print curvature_x
        curvature = np.c_[curvature_x, np.c_[curvature_y, curvature_z]]
        curvature = curvature + curvature.T
        return curvature

    def surface_normal(self, coords, delta=1.5):
        """Returns the sdf surface normal at the given coordinates by
        computing the tangent plane using SDF interpolation.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        delta : float
            A radius for collecting surface points near the target coords
            for calculating the surface normal.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3-dimensional ndarray that represents the surface normal.

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')

        # log warning if out of bounds access
        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        coords[0] = max(0, min(coords[0], self.dims_[0] - 1))
        coords[1] = max(0, min(coords[1], self.dims_[1] - 1))
        coords[2] = max(0, min(coords[2], self.dims_[2] - 1))
        index_coords = np.zeros(3)

        # check points on surface
        sdf_val = self[coords]
        if np.abs(sdf_val) >= self.surface_thresh_:
            logging.warning('Cannot compute normal. Point must be on surface')
            return None

        # collect all surface points within the delta sphere
        X = []
        d = np.zeros(3)
        dx = -delta
        while dx <= delta:
            dy = -delta
            while dy <= delta:
                dz = -delta
                while dz <= delta:
                    d = np.array([dx, dy, dz])
                    if dx != 0 or dy != 0 or dz != 0:
                        d = delta * d / np.linalg.norm(d)
                    index_coords[0] = coords[0] + d[0]
                    index_coords[1] = coords[1] + d[1]
                    index_coords[2] = coords[2] + d[2]
                    sdf_val = self[index_coords]
                    if np.abs(sdf_val) < self.surface_thresh_:
                        X.append([index_coords[0], index_coords[1], index_coords[2], sdf_val])
                    dz += delta
                dy += delta
            dx += delta

        # fit a plane to the surface points
        X.sort(key = lambda x: x[3])
        X = np.array(X)[:,:3]
        #X = X[:9,:] # take only the 9 'best' surface points for tangent plane calculation
        A = X - np.mean(X, axis=0)
        try:
            U, S, V = np.linalg.svd(A.T)
            n = U[:,2]
        except:
            logging.warning('Tangent plane does not exist. Returning None.')
            return None
        return n

    def surface_points(self, grid_basis=True):
        """Returns the points on the surface.

        Parameters
        ----------
        grid_basis : bool
            If False, the surface points are transformed to the world frame.
            If True (default), the surface points are left in grid coordinates.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of float
            The points on the surface and the signed distances at those points.
        """
        surface_points = np.where(np.abs(self.data_) < self.surface_thresh_)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_vals = self.data_[surface_points[:,0], surface_points[:,1], surface_points[:,2]]
        if not grid_basis:
            surface_points = self.transform_pt_grid_to_obj(surface_points.T)
            surface_points = surface_points.T

        return surface_points, surface_vals

    # TODO uses similarity tf
    def transform(self, tf, detailed = False):
        """
        Transform the grid by pose T and scale with canonical reference frame at the SDF center with axis alignment
        Params:
            (similarity transform 3d): similarity tf
            (bool): detailed - whether or not to do the dirty, fast method
        Returns:
            (SDF): new sdf with grid warped by T
        """
        # map all surface points to their new location
        tf = tf.inverse() # invert for correct lookups
        start_t = time.clock()
        num_pts = self.pts_.shape[0]
        pts_sdf = self.tf_grid_sdf_.apply(self.pts_.T)
        pts_sdf_tf = tf.apply(pts_sdf)
        pts_grid_tf = self.tf_sdf_grid_.apply(pts_sdf_tf)
        pts_tf = pts_grid_tf.T
        all_points_t = time.clock()

        # transform the center
        origin_sdf = self.tf_grid_sdf_.apply(self.origin_)
        origin_sdf_tf = tf.apply(origin_sdf)
        origin_tf = self.tf_sdf_grid_.apply(origin_sdf_tf)

        # rescale the resolution
        resolution_tf = tf.scale * self.resolution_
        origin_res_t = time.clock()

        # add each point to the new pose
        if detailed:
            sdf_data_tf = np.zeros([num_pts, 1])
            for i in range(num_pts):
                sdf_data_tf[i] = self[pts_tf[i,0], pts_tf[i,1], pts_tf[i,2]]
        else:
            pts_tf_round = np.round(pts_tf).astype(np.int64)

            # snap to closest boundary
            pts_tf_round[:,0] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,0]], axis=1)
            pts_tf_round[:,0] = np.min(np.c_[(self.dims_[0] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,0]], axis=1)

            pts_tf_round[:,1] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,1]], axis=1)
            pts_tf_round[:,1] = np.min(np.c_[(self.dims_[1] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,1]], axis=1)

            pts_tf_round[:,2] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,2]], axis=1)
            pts_tf_round[:,2] = np.min(np.c_[(self.dims_[2] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,2]], axis=1)

            sdf_data_tf = self.data_[pts_tf_round[:,0], pts_tf_round[:,1], pts_tf_round[:,2]]

        sdf_data_tf_grid = sdf_data_tf.reshape(self.dims_)
        tf_t = time.clock()

        logging.debug('Sdf3D: Time to transform coords: %f' %(all_points_t - start_t))
        logging.debug('Sdf3D: Time to transform origin: %f' %(origin_res_t - all_points_t))
        logging.debug('Sdf3D: Time to transfer sd: %f' %(tf_t - origin_res_t))
        tf.from_frame = self.tf_.to_frame
        tf.to_frame = self.tf_.to_frame
        return Sdf3D(sdf_data_tf_grid, origin_tf, resolution_tf, tf = tf.dot(self.tf_))

    # TODO uses similarity tf
    def transform_pt_obj_to_grid(self, x_sdf, direction = False):
        """ Converts a point in sdf coords to the grid basis. If direction then don't translate """
        return self.tf_sdf_grid_.apply(x_sdf, direction=direction)

    # TODO uses similarity tf
    def transform_pt_grid_to_obj(self, x_grid, direction = False):
        """ Converts a point in grid coords to the world basis. If direction then don't translate """
        return self.tf_grid_sdf_.apply(x_grid, direction=direction)

    # TODO this isn't used.
    def make_windows(self, W, S, target=False, filtering_function=crosses_threshold, threshold=.1):
        """
        Function for windowing the SDF grid
        Params:
            W: the side length of the window (currently assumed to be odd so centering makes sense)
            S: stride length between cubes (x axis "wraps" around)
            target: True for targetted windowing (filters for cubes containing both positive and negative values)
            filtering_function: function to filter windows out with
        Returns:
            ([np.array,...,np.array]): contains a list of numpy arrays, each of which is an unrolled window/cube.
                                       window order based on center coordinate (increasing order of x, y, then z)
        """
        windows = []
        window_center = (W-1)/2 #assuming odd window
        offset = 0 #parameter used to "wrap around" x / not start back at x=1 each time

        nx = self.dims_[0]
        ny = self.dims_[1]
        nz = self.dims_[2]
        newx = 2*window_center + nx
        newy = 2*window_center + ny
        newz = 2*window_center + nz
        padded_vals = np.zeros(newx*newy*newz)

        #padding the original values list
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    padded_coord = (i+window_center) + (j+window_center)*newx + (k+window_center)*newx*newy
                    orig_coord = i + j*nx + k*nx*ny
                    padded_vals[padded_coord] = self.values_in_order_[orig_coord]

        for z in range(window_center, nz + window_center, S):
            for y in range(window_center, ny + window_center, S):
                for x in range(window_center+offset, nx + window_center, S):
                        #print map(lambda x: x-window_center,[x,y,z])
                    new_window = np.zeros(W**3)
                    count = 0
                    for k in range(-window_center, window_center+1):
                        for j in range(-window_center, window_center+1):
                            for i in range(-window_center, window_center+1):
                                new_window[count] = padded_vals[(x+i) + (y+j)*newx + (z+k)*newx*newy]
                                count += 1
                    windows.append(new_window)
                offset = (x+S) - (nx+window_center)
        #print windows, len(windows), type(windows)
        if target:
            windows = filter(filtering_function(threshold), windows)
        return windows

    def add_to_nearpy_engine(self, engine):
        """Inserts the SDF into the provided nearpy Engine.
        Params:
            engine: nearpy.engine.Engine
        Returns: -
        """
        engine.store_vector(self.data_[:], self.file_name_)


    def query_nearpy_engine(self, engine):
        """
        Queries the provided nearpy Engine for the SDF closest to this one
        Params:
            engine: nearpy.engine.Engine
        Returns:
            (list (strings), list (tuples))
            list (strings): Names of the files that most closely match this one
            list (tuple): Additional information regarding the closest matches in (numpy.ndarray, string, numpy.float64) form:
                numpy.ndarray: the full vector of values of that match (equivalent to that SDF's "values_in_order_")
                string: the match's SDF's file name
                numpy.float64: the match's distance from this SDF
        """
        if self.feature_vector is None:
            to_query = self.data_[:]
        else:
            to_query = self.feature_vector
        results = engine.neighbours(to_query)
        file_names = [i[1] for i in results]
        return file_names, results


class Sdf2D(Sdf):
    def __init__(self, sdf_data, origin = np.array([0,0]), resolution = 1.0, pose = tfx.identity_tf(from_frame="world"), scale = 1.0):
        self.data_ = sdf_data
        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape
        self.pose_ = pose
        self.scale_ = scale

        self.surface_thresh_ = self.resolution_ * np.sqrt(2) / 2 # resolution is max dist from surface when surf is orthogonal to diagonal grid cells
        self.center_ = np.array([self.dims_[0] / 2, self.dims_[1] / 2])

        self._compute_flat_indices()
        self._compute_gradients()

        self.feature_vector_ = None #Kmeans feature representation

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind] = np.indices(self.dims_)
        self.pts_ = np.c_[x_ind.flatten().T, y_ind.flatten().T];

    def __getitem__(self, coords):
        """
        Returns the signed distance at the given coordinates, interpolating if necessary
        Params: numpy 3 array
        Returns:
            float: the signed distance and the given coors (interpolated)
        """
        # if len(coords) != 2:
        #     raise IndexError('Indexing must be 2 dimensional')

        if self.is_out_of_bounds(coords):
            logging.debug('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        new_coords = np.zeros([2,])
        new_coords[0] = max(0, min(coords[0], self.dims_[0]-1))
        new_coords[1] = max(0, min(coords[1], self.dims_[1]-1))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int:
            new_coords = new_coords.astype(np.int)
            return self.data_[new_coords[0], new_coords[1]]

        # otherwise interpolate
        min_coords = np.floor(new_coords)
        max_coords = np.ceil(new_coords)
        points = np.array([[min_coords[0], min_coords[1]],
                           [max_coords[0], min_coords[1]],
                           [min_coords[0], max_coords[1]],
                           [max_coords[0], max_coords[1]]])

        num_interpolants= 4
        values = np.zeros([num_interpolants,])
        weights = np.ones([num_interpolants,])
        for i in range(num_interpolants):
            p = points[i,:].astype(np.int)
            values[i] = self.data_[p[0], p[1]]
            dist = np.linalg.norm(new_coords - p.T)
            if dist > 0:
                weights[i] = 1.0 / dist

        if np.sum(weights) == 0:
            weights = np.ones([num_interpolants,])
        weights = weights / np.sum(weights)
        return weights.dot(values)

    def surface_points(self, surf_thresh=None):
        """
        Returns the points on the surface
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        if surf_thresh is None:
            surf_thresh = self.surface_thresh_
        surface_points = np.where(np.abs(self.data_) < surf_thresh)
        x = surface_points[0]
        y = surface_points[1]
        surface_points = np.c_[x, y]
        surface_vals = self.data_[surface_points[:,0], surface_points[:,1]]
        return surface_points, surface_vals

    def surface_image_thresh(self, alpha = 0.5, scale = 4):
        """
        Returns an image that is zero on the shape surface and one otherwise
        """
        surface_points, surface_vals = self.surface_points(self.surface_thresh_)
        surface_image = np.zeros(self.dims_).astype(np.uint8)
        surface_image[surface_points[:,0], surface_points[:,1]] = MAX_CHAR

        # laplacian
        surface_image = np.array(Image.fromarray(surface_image).resize((scale*self.dims_[0], scale*self.dims_[1]), Image.ANTIALIAS))
        surface_image = scipy.ndimage.gaussian_filter(surface_image, 2)
        filter_blurred_l = scipy.ndimage.gaussian_filter(surface_image, 0.1)
        surface_image = surface_image + alpha * (surface_image - filter_blurred_l)

        # take only points higher than a certain value
        surface_image = MAX_CHAR * (surface_image > 70)
        return surface_image

    def transform(self, T, scale = 1.0):
        """
        Transform the grid by pose T and scale with canonical reference frame at the SDF center with axis alignment
        Params:
            (tfx pose): Pose T
            (float): scale of transform
        Returns:
            (SDF): new sdf with grid warped by T
        """
        # map all surface points to their new location
        num_pts = self.pts_.shape[0]
        pts_centered = self.pts_ - self.center_
        pts_homog = np.r_[pts_centered.T, np.r_[np.zeros([1, num_pts]), np.ones([1, num_pts])]]
        pts_homog_tf = T.matrix.dot(pts_homog)
        pts_tf_centered = (1.0 / scale) * pts_homog_tf[0:2,:].T
        pts_tf = pts_tf_centered + self.center_

        # add each point to the new pose
        sdf_data_tf = np.zeros([num_pts, 1])
        for i in range(num_pts):
            sdf_data_tf[i] = self[pts_tf[i,0], pts_tf[i,1]]
        sdf_data_tf_grid = sdf_data_tf.reshape(self.dims_)

        return Sdf2D(sdf_data_tf_grid, pose = T * self.pose_)

    def set_feature_vector(self, vector):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor
        """
        self.feature_vector_ = vector

    def feature_vector(self):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor
        """
        return self.feature_vector_


    def add_to_nearpy_engine(self, engine):
        """
        Inserts the SDF into the provided nearpy Engine
        Params:
            engine: nearpy.engine.Engine
        Returns: -
        """
        if self.feature_vector_ is None:
            to_add = self.data_[:]
        else:
            to_add = self.feature_vector

        engine.store_vector(to_add,self.file_name_)


    def query_nearpy_engine(self, engine):
        """
        Queries the provided nearpy Engine for the SDF closest to this one
        Params:
            engine: nearpy.engine.Engine
        Returns:
            (list (strings), list (tuples))
            list (strings): Names of the files that most closely match this one
            list (tuple): Additional information regarding the closest matches in (numpy.ndarray, string, numpy.float64) form:
                numpy.ndarray: the full vector of values of that match (equivalent to that SDF's "values_in_order_")
                string: the match's SDF's file name
                numpy.float64: the match's distance from this SDF
        """
        if self.feature_vector is None:
            to_query = self.data_[:]
        else:
            to_query = self.feature_vector
        results = engine.neighbours(to_query)
        file_names = [i[1] for i in results]
        return file_names, results

    def send_to_matlab(self, out_file):
        """
        Saves the SDF's coordinate and value information to the provided matlab workspace
        Params:
            out_file: string
        Returns: -
        """
        # TODO: fix this
#        scipy.io.savemat(out_file, mdict={'X':self.xlist_, 'Y': self.ylist_, 'Z': self.zlist_, 'vals': self.values_in_order_})
        logging.info("SDF information saved to %s" % out_file)

    def imshow(self):
        """
        Displays the SDF as an image
        """
        plt.imshow(self.data_ < 0, cmap=plt.get_cmap('Greys'))

    def vis_surface(self):
        """
        Displays the SDF surface image
        """
        plt.imshow(self.surface_image_thresh(), cmap=plt.get_cmap('Greys'))

def crosses_threshold(threshold):
    def crosses(elems):
        """
        Function to determine if a np array has values both above and below a threshold (moved edge). Generalized "has positive and negative" function.
        For use with filter(-,-).
        Params:
            elems: np array of numbers
            threshold: any number used as a threshold
        Returns:
            (bool): True if elems has both negative and positive numbers, False otherwise
        """
        return (elems>threshold).any() and (elems<threshold).any()
    return crosses

def random_frame(length=10):
    frame = 'obj_'
    for i in range(length):
        frame = frame.join(random.choice(string.ascii_uppercase + string.digits))
    return frame

def find_zero_crossing_linear(x1, y1, x2, y2):
    """ Find zero crossing using linear approximation"""
    # NOTE: use sparingly, approximations can be shoddy
    d = x2 - x1
    t1 = 0
    t2 = np.linalg.norm(d)
    v = d / t2

    m = (y2 - y1) / (t2 - t1)
    b = y1
    t_zc = -b / m
    x_zc = x1 + t_zc * v
    return x_zc

def find_zero_crossing_quadratic(x1, y1, x2, y2, x3, y3):
    """ Find zero crossing using quadratic approximation along 1d line"""
    # compute coords along 1d line
    v = x2 - x1
    v = v / np.linalg.norm(v)
    if v[v!=0].shape[0] == 0:
        logging.error('Difference is 0. Probably a bug')

    t1 = 0
    t2 = (x2 - x1)[v!=0] / v[v!=0]
    t2 = t2[0]
    t3 = (x3 - x1)[v!=0] / v[v!=0]
    t3 = t3[0]

    # solve for quad approx
    x1_row = np.array([t1**2, t1, 1])
    x2_row = np.array([t2**2, t2, 1])
    x3_row = np.array([t3**2, t3, 1])
    X = np.array([x1_row, x2_row, x3_row])
    y_vec = np.array([y1, y2, y3])
    try:
        w = np.linalg.solve(X, y_vec)
    except np.linalg.LinAlgError:
        logging.error('Singular matrix. Probably a bug')

    # get positive roots
    possible_t = np.roots(w)
    t_zc = None
    for i in range(possible_t.shape[0]):
        if possible_t[i] >= 0 and possible_t[i] <= 10 and not np.iscomplex(possible_t[i]):
            t_zc = possible_t[i]

    # if no positive roots find min
    if t_zc is None:
        t_zc = -w[1] / (2 * w[0])

    eps = 1.0
    if t_zc < -eps or t_zc > eps:
        return None

    x_zc = x1 + t_zc * v
    return x_zc

def test_function():
    test_sdf = "aunt_jemima_original_syrup/processed/textured_meshes/optimized_tsdf_texture_mapped_mesh.sdf"
    matlab_file = "data.mat"
    teatime = SDF(test_sdf)
    print "Done processing %s" % test_sdf
    print "Dimension: %d, x: %d, y: %d, z: %d" % teatime.dimensions()
    print "Origin: (%f, %f, %f)" % teatime.origin()
    print "Spacing: %f" % teatime.spacing()
    print "Data Matrix: \n", teatime.data()

    #Testing LSH by just inserting the same sdf file twice, then querying it
    dimension = teatime.dimensions()[0]
    rbp = RandomBinaryProjections('rbp',10)
    engine = Engine(dimension, lshashes=[rbp])
    teatime.add_to_nearpy_engine(engine)
    teatime.add_to_nearpy_engine(engine)
    print "Query results: \n", teatime.query_nearpy_engine(engine)

    teatime.send_to_matlab(matlab_file)
    teatime.make_plot()

def test_3d_transform():
    np.random.seed(100)
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    s = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = s.read()

    # transform
    pose_mat = np.matrix([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0,0,1]])
    tf = tfx.transform(pose_mat, from_frame='world')
    tf = tfx.random_tf()
    tf.position = 0.01 * np.random.rand(3)

    start_t = time.clock()
    s_tf = stf.SimilarityTransform3D(tf, scale = 1.2)
    sdf_tf = sdf_3d.transform(s_tf)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('3D Transform took %f sec' %(duration))
    logging.info('Transformed resolution %f' %(sdf_tf.resolution))

    start_t = time.clock()
    sdf_tf_d = sdf_3d.transform(s_tf, detailed = True)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('Detailed 3D Transform took %f sec' %(duration))
    logging.info('Transformed detailed resolution %f' %(sdf_tf_d.resolution))

    # display
    plt.figure()
    sdf_3d.scatter()
    plt.title('Original')

    plt.figure()
    sdf_tf.scatter()
    plt.title('Transformed')

    plt.figure()
    sdf_tf_d.scatter()
    plt.title('Detailed Transformed')
    plt.show()

def test_2d_transform():
    sdf_2d_file_name = 'data/test/sdf/medium_black_spring_clamp_optimized_poisson_texture_mapped_mesh_clean_0.csv'
    sf2 = sf.SdfFile(sdf_2d_file_name)
    sdf_2d = sf2.read()

    # transform
    pose_mat = np.matrix([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0,0,1]])
    tf = tfx.transform(pose_mat, from_frame='world')

    start_t = time.clock()
    sdf_tf = sdf_2d.transform(tf, scale = 1.5)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('2D Transform took %f sec' %(duration))

    # display
    plt.figure()
    plt.subplot(1,2,1)
    sdf_2d.imshow()
    plt.title('Original')

    plt.subplot(1,2,2)
    sdf_tf.imshow()
    plt.title('Transformed')

    plt.show()

def test_quad_zc():
    x1 = np.array([1, 1, 1])
    x2 = np.array([0, 0, 0])
    x3 = np.array([-1, -1, -1])
    y1 = 3
    y2 = 1
    y3 = -1.5
    x_zc = find_zero_crossing(x1, y1, x2, y2, x3, y3)
    true_x_zc = -0.4244289 * np.ones(3)
    assert(np.linalg.norm(x_zc - true_x_zc) < 1e-2)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
#    test_2d_transform()
    test_3d_transform()

