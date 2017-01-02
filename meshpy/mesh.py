"""
Encapsulates mesh for grasping operations
Author: Jeff Mahler
"""
import math
import Queue
import os
import sys

import numpy as np
import scipy.spatial as ss
import sklearn.decomposition

from core import RigidTransform, PointCloud

import obj_file
import stable_pose as sp

class Mesh3D(object):
    """A triangular mesh for a three-dimensional shape representation.

    Attributes
    ----------
    vertices : :obj:`numpy.ndarray` of float
        A #verts by 3 array, where each row contains an ordered
        [x,y,z] set that describes one vertex.
    triangles : :obj:`numpy.ndarray`  of int
        A #tris by 3 array, where each row contains indices of vertices in
        the `vertices` array that are part of the triangle.
    normals : :obj:`numpy.ndarray` of float
        A #normals by 3 array, where each row contains a normalized
        vector. This list should contain one norm per vertex.
    density : float
        The density of the mesh.
    center_of_mass : :obj:`numpy.ndarray` of float
        The 3D location of the mesh's center of mass.
    mass : float
        The mass of the mesh (read-only).
    inertia : :obj:`numpy.ndarray` of float
        The 3x3 inertial matrix of the mesh (read-only).
    bb_center : :obj:`numpy.ndarray` of float
        The 3D location of the center of the mesh's minimal bounding box
        (read-only).
    centroid : :obj:`numpy.ndarray` of float
        The 3D location of the mesh's vertex mean (read-only).
    """

    ScalingTypeMin = 0
    ScalingTypeMed = 1
    ScalingTypeMax = 2
    ScalingTypeRelative = 3
    ScalingTypeDiag = 4
    OBJ_EXT = '.obj'
    PROC_TAG = '_proc'
    C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])

    def __init__(self, vertices, triangles, normals=None,
                 density=1.0, center_of_mass=None, uniform_com=False):
        """Construct a 3D triangular mesh.

        Parameters
        ----------
        vertices : :obj:`numpy.ndarray` of float
            A #verts by 3 array, where each row contains an ordered
            [x,y,z] set that describes one vertex.
        triangles : :obj:`numpy.ndarray`  of int
            A #tris by 3 array, where each row contains indices of vertices in
            the `vertices` array that are part of the triangle.
        normals : :obj:`numpy.ndarray` of float
            A #normals by 3 array, where each row contains a normalized
            vector. This list should contain one norm per vertex.
        density : float
            The density of the mesh.
        center_of_mass : :obj:`numpy.ndarray` of float
            The 3D location of the mesh's center of mass.
        uniform_com : bool
            Whether or not to assume a uniform mass density for center of mass comp
        """
        if vertices is not None:
            vertices = np.array(vertices)
        self.vertices_ = vertices

        if triangles is not None:
            triangles = np.array(triangles)
        self.triangles_ = triangles

        if normals is not None:
            normals = np.array(normals)
            if normals.shape[0] == 3:
                normals = normals.T
        self.normals_ = normals

        self.density_ = density

        self.center_of_mass_ = center_of_mass

        # Read-Only parameter initialization
        self.mass_ = None
        self.inertia_ = None
        self.bb_center_ = self._compute_bb_center() 
        self.centroid_ = self._compute_centroid()

        if self.center_of_mass_ is None:
            if uniform_com:
                self.center_of_mass_ = self._compute_com_uniform()
            else:
                self.center_of_mass_ = self.bb_center_


    ##################################################################
    # Properties
    ##################################################################

    #=============================================
    # Read-Write Properties
    #=============================================
    @property
    def vertices(self):
        """:obj:`numpy.ndarray` of float : A #verts by 3 array,
        where each row contains an ordered
        [x,y,z] set that describes one vertex.
        """
        return self.vertices_

    @vertices.setter
    def vertices(self, v):
        self.vertices_ = np.array(v)
        self.mass_ = None
        self.inertia_ = None
        self.bb_center_ = self._compute_bb_center()
        self.centroid_ = self._compute_centroid()

    @property
    def triangles(self):
        """:obj:`numpy.ndarray` of int : A #tris by 3 array,
        where each row contains indices of vertices in
        the `vertices` array that are part of the triangle.
        """
        return self.triangles_

    @triangles.setter
    def triangles(self, t):
        self.triangles_ = np.array(t)
        self.mass_ = None
        self.inertia_ = None

    @property
    def normals(self):
        """:obj:`numpy.ndarray` of float :
        A #normals by 3 array, where each row contains a normalized
        vector. This list should contain one norm per vertex.
        """
        return self.normals_

    @normals.setter
    def normals(self, n):
        self.normals_ = np.array(n)

    @property
    def density(self):
        """float : The density of the mesh.
        """
        return self.density_

    @density.setter
    def density(self, d):
        self.density_ = d
        self.mass_ = None
        self.inertia_ = None

    @property
    def center_of_mass(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the mesh's center of mass.
        """
        return self.center_of_mass_

    @center_of_mass.setter
    def center_of_mass(self, com):
        self.center_of_mass_ = com
        self.inertia_ = None

    #=============================================
    # Read-Only Properties
    #=============================================
    @property
    def mass(self):
        """float : The mass of the mesh (read-only).
        """
        if self.mass_ is None:
            self.mass_ = self._compute_mass()
        return self.mass_

    @property
    def inertia(self):
        """:obj:`numpy.ndarray` of float :
        The 3x3 inertial matrix of the mesh (read-only).
        """
        if self.inertia_ is None:
            self.inertia_ = self._compute_inertia()
        return self.inertia_

    @property
    def bb_center(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the center of the mesh's minimal bounding box
        (read-only).
        """
        return self.bb_center_

    @property
    def centroid(self):
        """:obj:`numpy.ndarray` of float :
        The 3D location of the mesh's vertex mean (read-only).
        """
        return self.centroid_

    ##################################################################
    # Public Class Methods
    ##################################################################

    def min_coords(self):
        """Returns the minimum coordinates of the mesh.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the minimal
            x, y, and z coordinates represented in the mesh.
        """
        return np.min(self.vertices_, axis=0)

    def max_coords(self):
        """Returns the maximum coordinates of the mesh.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the minimal
            x, y, and z coordinates represented in the mesh.
        """
        return np.max(self.vertices_, axis=0)

    def bounding_box(self):
        """Returns the mesh's bounding box corners.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            A 2-tuple of 3-ndarrays of floats. The first 3-array
            contains the vertex of the smallest corner of the bounding box,
            and the second 3-array contains the largest corner of the bounding
            box.
        """
        return self.min_coords(), self.max_coords()

    def bounding_box_mesh(self):
        """Returns the mesh bounding box as a mesh.

        Returns
        -------
        :obj:`Mesh3D`
            A Mesh3D representation of the mesh's bounding box.
        """
        min_vert, max_vert = self.bounding_box()
        xs, ys, zs = zip(max_vert, min_vert)
        vertices = []
        for x in xs:
            for y in ys:
                for z in zs:
                    vertices.append([x, y, z])
        triangles = (np.array([
            [5, 7, 3], [5, 3, 1],
            [2, 4, 8], [2, 8, 6],
            [6, 8, 7], [6, 7, 5],
            [1, 3, 4], [1, 4, 2],
            [6, 5, 1], [6, 1, 2],
            [7, 8, 4], [7, 4, 3],
        ]) - 1)
        return Mesh3D(vertices, triangles)

    def principal_dims(self):
        """Returns the maximal span of the mesh's coordinates.

        The maximal span is the maximum coordinate value minus
        the minimal coordinate value in each principal axis.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the maximal
            x, y, and z spans of the mesh.
        """
        return self.max_coords() - self.min_coords()

    def support(self, direction):
        """Returns the support function in the given direction

        Parameters
        ----------
        direction : :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that is a unit vector in
            the direction of the desired support.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-ndarray of floats that represents the support.
        """
        ip = self.vertices_.dot(direction)
        index = np.where(ip == np.max(ip))[0][0]
        x0 = self.vertices_[index,:]
        n = direction
        com_proj = x0.dot(n) * n
        return com_proj

    def tri_centers(self):
        """Returns an array of the triangle centers as 3D points.

        Returns
        -------
        :obj:`numpy.ndarray` of :obj:`numpy.ndarray` of float
            An ndarray of 3-ndarrays of floats, where each 3-ndarray
            represents the 3D point at the center of the corresponding
            mesh triangle.
        """
        centers = []
        for tri in self.triangles_:
            centers.append(self._center_of_tri(tri))
        return np.array(centers)

    def tri_normals(self, align_to_hull=False):
        """Returns a list of the triangle normals.

        Parameters
        ----------
        align_to_hull : bool
            If true, we re-orient the normals to point outward from
            the mesh by using the convex hull.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A #triangles by 3 array of floats, where each 3-ndarray
            represents the 3D normal vector of the corresponding triangle.
        """
        # compute normals
        v0 = self.vertices_[self.triangles_[:,0],:]
        v1 = self.vertices_[self.triangles_[:,1],:]
        v2 = self.vertices_[self.triangles_[:,2],:]
        n = np.cross(v1 - v0, v2 - v0)
        normals = n / np.tile(np.linalg.norm(n, axis=1)[:,np.newaxis], [1,3])

        # reverse normal based on alignment with convex hull
        if align_to_hull:
            tri_centers = self.tri_centers()
            hull = ss.ConvexHull(tri_centers)
            hull_tris = hull.simplices
            hull_vertex_ind = hull_tris[0][0]
            hull_vertex = tri_centers[hull_vertex_ind]
            hull_vertex_normal = normals[hull_vertex_ind]
            v = hull_vertex.reshape([1,3])
            n = hull_vertex_normal
            ip = (tri_centers - np.tile(hull_vertex,
                  [tri_centers.shape[0], 1])).dot(n)
            if ip[0] > 0:
                normals = -normals
        return normals

    def surface_area(self):
        """Return the surface area of the mesh.

        Returns
        -------
        float
            The surface area of the mesh.
        """
        area = 0.0
        for tri in self.triangles:
            tri_area = self._area_of_tri(tri)
            area += tri_area
        return area

    def total_volume(self):
        """Return the total volume of the mesh.

        Returns
        -------
        float
            The total volume of the mesh.
        """
        total_volume = 0
        for tri in self.triangles_:
            volume = self._signed_volume_of_tri(tri)
            total_volume = total_volume + volume

        # Correct for flipped triangles
        if total_volume < 0:
            total_volume = -total_volume
        return total_volume

    def covariance(self):
        """Return the total covariance of the mesh's triangles.

        Returns
        -------
        float
            The total covariance of the mesh's triangles.
        """
        C_sum = np.zeros([3,3])
        for tri in self.triangles_:
            C = self._covariance_of_tri(tri)
            C_sum = C_sum + C
        return C_sum

    def remove_bad_tris(self):
        """Remove triangles with out-of-bounds vertices from the mesh.
        """
        new_tris = []
        num_v = self.vertices_.shape[0]
        for t in self.triangles_:
            if (t[0] >= 0 and t[0] < num_v and
                t[1] >= 0 and t[1] < num_v and
                t[2] >= 0 and t[2] < num_v):
                new_tris.append(t)
        self.triangles = np.array(new_tris)

    def remove_unreferenced_vertices(self):
        """Remove any vertices that are not part of a triangular face.

        Note
        ----
        This method will fail if any bad triangles are present, so run
        remove_bad_tris() first if you're unsure if bad triangles are present.

        Returns
        -------
        bool
            Returns True if vertices were removed, False otherwise.

        """
        num_v = self.vertices_.shape[0]

        # Fill in a 1 for each referenced vertex
        reffed_array = np.zeros([num_v, 1])
        for f in self.triangles_:
            reffed_array[f[0]] = 1
            reffed_array[f[1]] = 1
            reffed_array[f[2]] = 1

        # Trim out vertices that are not referenced
        reffed_v_old_ind = np.where(reffed_array == 1)
        reffed_v_old_ind = reffed_v_old_ind[0]

        # Count number of referenced vertices before each index
        reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1

        try:
            self.vertices = self.vertices_[reffed_v_old_ind, :]
            if self.normals is not None:
                self.normals = self.normals[reffed_v_old_ind, :]
        except IndexError:
            return False

        # create new face indices
        new_triangles = []
        for f in self.triangles_:
            new_triangles.append([reffed_v_new_ind[f[0]],
                                  reffed_v_new_ind[f[1]],
                                  reffed_v_new_ind[f[2]]])
        self.triangles = np.array(new_triangles)
        return True

    def center_vertices_avg(self):
        """Center the mesh's vertices at the centroid.

        This shifts the mesh without rotating it so that
        the centroid (mean) of all vertices is at the origin.
        """
        centroid = np.mean(self.vertices_, axis = 0)
        self.vertices = self.vertices_ - centroid

    def center_vertices_bb(self):
        """Center the mesh's vertices at the center of its bounding box.

        This shifts the mesh without rotating it so that
        the center of its bounding box is at the origin.
        """
        min_vertex = self.min_coords()
        max_vertex = self.max_coords()
        center = (max_vertex + min_vertex) / 2
        self.vertices = self.vertices_ - center

    def normalize_vertices(self):
        """Normalize the mesh's orientation along its principal axes.

        Transforms the vertices and normals of the mesh
        such that the origin of the resulting mesh's coordinate frame
        is at the center of the bounding box and the principal axes (as determined
        from PCA) are aligned with the vertical Z, Y, and X axes in that order.
        """

        self.center_vertices_bb()

        # Find principal axes
        pca = sklearn.decomposition.PCA(n_components = 3)
        pca.fit(self.vertices_)

        # Count num vertices on side of origin wrt principal axes
        # to determine correct orientation
        comp_array = pca.components_
        norm_proj = self.vertices_.dot(comp_array.T)
        opposite_aligned = np.sum(norm_proj < 0, axis = 0)
        same_aligned = np.sum(norm_proj >= 0, axis = 0)

        # create rotation from principal axes to standard basis
        z_axis = comp_array[0,:]
        y_axis = comp_array[1,:]
        if opposite_aligned[2] > same_aligned[2]:
            z_axis = -z_axis
        if opposite_aligned[1] > same_aligned[1]:
            y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
        R_pc_obj = np.c_[x_axis, y_axis, z_axis]

        # rotate vertices, normals and reassign to the mesh
        self.vertices = (R_pc_obj.T.dot(self.vertices.T)).T
        self.center_vertices_bb()

        # TODO JEFF LOOK HERE (BUG IN INITIAL CODE FROM MESHPROCESSOR)
        if self.normals_ is not None:
            self.normals = (R_pc_obj.T.dot(self.normals.T)).T

    def compute_vertex_normals(self):
        """ Get normals from triangles"""
        normals = []
        for i in range(len(self.vertices)):
            inds = np.where(self.triangles == i)
            first_tri = self.triangles[inds[0][0],:]
            t = self.vertices[first_tri, :]
            v0 = t[1,:] - t[0,:]
            v1 = t[2,:] - t[0,:]
            v0 = v0 / np.linalg.norm(v0)
            v1 = v1 / np.linalg.norm(v1)
            n = np.cross(v0, v1)
            n = n / np.linalg.norm(n)
            normals.append(n.tolist())

        # Reverse normals based on alignment with convex hull
        hull = ss.ConvexHull(self.vertices_)
        hull_tris = hull.simplices.tolist()
        hull_vertex_ind = hull_tris[0][0]
        hull_vertex = self.vertices[hull_vertex_ind]
        hull_vertex_normal = normals[hull_vertex_ind]
        v = np.array(hull_vertex).reshape([1,3])
        n = np.array(hull_vertex_normal)
        ip = (self.vertices - np.tile(hull_vertex, [self.vertices.shape[0], 1])).dot(n)
        if ip[0] > 0:
            normals = [[-n[0], -n[1], -n[2]] for n in normals]
        self.normals = normals

    def scale_principal_eigenvalues(self, new_evals):
        self.normalize_vertices()

        pca = sklearn.decomposition.PCA(n_components = 3)
        pca.fit(self.vertices_)

        evals = pca.explained_variance_
        if len(new_evals) == 3:
            self.vertices[:,0] *= new_evals[2]/np.sqrt(evals[2])
            self.vertices[:,1] *= new_evals[1]/np.sqrt(evals[1])
            self.vertices[:,2] *= new_evals[0]/np.sqrt(evals[0])
        elif len(new_evals) == 2:
            self.vertices[:,1] *= new_evals[1]/np.sqrt(evals[1])
            self.vertices[:,2] *= new_evals[0]/np.sqrt(evals[0])
        elif len(new_evals) == 1:
            self.vertices[:,0] *= new_evals[0]/np.sqrt(evals[0])
            self.vertices[:,1] *= new_evals[0]/np.sqrt(evals[0])
            self.vertices[:,2] *= new_evals[0]/np.sqrt(evals[0])
        self.center_vertices_bb()
        return evals

    def copy(self):
        """Return a copy of the mesh.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        return Mesh3D(np.copy(self.vertices_), np.copy(self.triangles_))

    def subdivide(self, min_tri_length = None):
        """Return a copy of the mesh that has been subdivided by one iteration.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        new_mesh = self.copy()
        new_vertices = new_mesh.vertices.tolist()
        old_triangles = new_mesh.triangles.tolist()

        new_triangles = []
        triangle_index_mapping = {}
        tri_queue = Queue.Queue()

        for j, triangle in enumerate(old_triangles):
            tri_queue.put((j, triangle))
            triangle_index_mapping[j] = []

        while not tri_queue.empty():
            tri_index_pair = tri_queue.get()
            j = tri_index_pair[0]
            triangle = tri_index_pair[1]

            if (min_tri_length is None or
                Mesh3D._max_edge_length(triangle, new_vertices) > min_tri_length):
                t_vertices = np.array([new_vertices[i] for i in triangle])
                edge01 = 0.5 * (t_vertices[0,:] + t_vertices[1,:])
                edge12 = 0.5 * (t_vertices[1,:] + t_vertices[2,:])
                edge02 = 0.5 * (t_vertices[0,:] + t_vertices[2,:])

                i_01 = len(new_vertices)
                i_12 = len(new_vertices)+1
                i_02 = len(new_vertices)+2
                new_vertices.append(edge01)
                new_vertices.append(edge12)
                new_vertices.append(edge02)

                for triplet in [[triangle[0], i_01, i_02],
                                [triangle[1], i_12, i_01],
                                [triangle[2], i_02, i_12],
                                [i_01, i_12, i_02]]:
                    tri_queue.put((j, triplet))

            else:
                new_triangles.append(triangle)
                triangle_index_mapping[j].append(len(new_triangles)-1)

        new_mesh.vertices = new_vertices
        new_mesh.triangles = new_triangles
        return new_mesh

    def transform(self, T):
        """Return a copy of the mesh that has been transformed by T.

        Parameters
        ----------
        T : :obj:`RigidTransform`
            The RigidTransform by which the mesh is transformed.

        Note
        ----
        This method only copies the vertices and triangles of the mesh.
        """
        vertex_cloud = PointCloud(self.vertices_.T, frame=T.from_frame)
        vertex_cloud_tf = T * vertex_cloud
        vertices = vertex_cloud_tf.data.T
        return Mesh3D(np.copy(vertices), np.copy(self.triangles))

    def random_points(self, n_points):
        """Generate uniformly random points on the surface of the mesh.

        Parameters
        ----------
        n_points : int
            The number of random points to generate.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A n_points by 3 ndarray that contains the sampled 3D points.
        """
        probs = self._tri_area_percentages()
        tri_inds = np.random.choice(range(len(probs)), n_points, p=probs)
        points = []
        for tri_ind in tri_inds:
            tri = self.triangles[tri_ind]
            points.append(self._rand_point_on_tri(tri))
        return np.array(points)

    def ray_intersections(self, ray, point, distance):
        """Returns a list containing the indices of the triangles that
        are intersected by the given ray emanating from the given point
        within some distance.
        """
        ray = ray / np.linalg.norm(ray)
        norms = self.tri_normals()
        tri_point_pairs = []
        for i, tri in enumerate(self.triangles):
            if np.dot(ray, norms[i]) == 0.0:
                continue
            t = -1 * np.dot((point - self.vertices[tri[0]]), norms[i]) / (np.dot(ray, norms[i]))
            if (t > 0 and t <= distance):
                contact_point = point + t * ray
                tri_verts = [self.vertices[j] for j in tri]
                if Mesh3D._point_in_tri(tri_verts, contact_point):
                    tri_point_pairs.append((i, contact_point))
        return tri_point_pairs

    def get_T_surface_obj(self, T_surface_ori_obj, delta=0.0):
        """ Gets the transformation that puts the object resting exactly on
        the z=delta plane

        Parameters
        ----------
        T_surface_ori_obj : :obj:`RigidTransform`
            The RigidTransform by which the mesh is transformed.
        delta : float
            Z-coordinate to rest the mesh on

        Note
        ----
        This method copies the vertices and triangles of the mesh.
        """
        obj_tf = self.transform(T_surface_ori_obj)
        mn, mx = obj_tf.bounding_box()

        z=mn[2]
        x0 = np.array([0,0,-z+delta])

        T_surface_obj = RigidTransform(rotation=T_surface_ori_obj.rotation,
                                       translation=x0, from_frame='obj',
                                       to_frame='surface')
        return T_surface_obj

    def rescale_dimension(self, scale, scaling_type=ScalingTypeMin):
        """Rescales the vertex coordinates to scale using the given scaling_type.

        Parameters
        ----------
        scale : float
            The desired scaling factor of the selected dimension, if scaling_type
            is ScalingTypeMin, ScalingTypeMed, ScalingTypeMax, or
            ScalingTypeDiag. Otherwise, the overall scaling factor.

        scaling_type : int
            One of ScalingTypeMin, ScalingTypeMed, ScalingTypeMax,
            ScalingTypeRelative, or ScalingTypeDiag.
            ScalingTypeMin scales the smallest vertex extent (X, Y, or Z)
            by scale, ScalingTypeMed scales the median vertex extent, and
            ScalingTypeMax scales the maximum vertex extent. ScalingTypeDiag
            scales the bounding box diagonal (divided by three), and
            ScalingTypeRelative provides absolute scaling.
        """
        vertex_extent = self.principal_dims()

        # Find minimal dimension
        relative_scale = 1.0
        if scaling_type == Mesh3D.ScalingTypeMin:
            dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeMed:
            dim = np.where(vertex_extent == np.med(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeMax:
            dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif scaling_type == Mesh3D.ScalingTypeRelative:
            relative_scale = 1.0
        elif scaling_type == Mesh3D.ScalingTypeDiag:
            diag = np.linalg.norm(vertex_extent)
            relative_scale = diag / 3.0 # make the gripper size exactly one third of the diagonal

        # Compute scale factor and rescale vertices
        scale_factor = scale / relative_scale
        self.vertices_ = scale_factor * self.vertices_

    def rescale(self, scale_factor):
        """Rescales the vertex coordinates by scale_factor.

        Parameters
        ----------
        scale_factor : float
            The desired scale factor for the mesh's vertices.
        """
        self.vertices = scale_factor * self.vertices_

    def convex_hull(self):
        """Return a 3D mesh that represents the convex hull of the mesh.
        """
        hull = ss.ConvexHull(self.vertices_)
        hull_tris = hull.simplices
        # TODO do normals properly...
        cvh_mesh = Mesh3D(np.copy(self.vertices_), np.copy(hull_tris))#, self.normals_)
        cvh_mesh.remove_unreferenced_vertices()
        return cvh_mesh

    def stable_poses(self):
        """Computes all valid StablePose objects for the mesh.

        Returns
        -------
        :obj:`list` of :obj:`StablePose`
            A list of StablePose objects for the mesh.
        """
        cm = self.center_of_mass
        cvh_mesh = self.convex_hull()

        cvh_tris = cvh_mesh.triangles
        cvh_verts  = cvh_mesh.vertices

        edge_to_faces = {} # Mapping from Edge objects to adjacent triangle lists
        tri_to_vert = {}   # Mapping from Triangle tuples to Vertex objects

        # Create a map from edges to bordering faces and from
        # faces to Vertex objects.
        for tri in cvh_tris:
            tri_verts = [cvh_verts[i] for i in tri]
            s1 = Mesh3D._Segment(tri_verts[0], tri_verts[1])
            s2 = Mesh3D._Segment(tri_verts[0], tri_verts[2])
            s3 = Mesh3D._Segment(tri_verts[1], tri_verts[2])
            for seg in [s1, s2, s3]:
                k = seg.tup
                if k in edge_to_faces:
                    edge_to_faces[k] += [tri]
                else:
                    edge_to_faces[k] = [tri]

            p = self._compute_proj_area(tri_verts) / (4 * math.pi)
            tri_to_vert[tuple(tri)] = Mesh3D._GraphVertex(p, tri)

        # determining if landing on a given face implies toppling, and initializes a directed acyclic graph
        # a directed edge between two graph nodes implies that landing on one face will lead to toppling onto its successor
        # an outdegree of 0 for any graph node implies it is a sink (the object will come to rest if it topples to this face)
        for tri in cvh_tris:
            tri_verts = [cvh_verts[i] for i in tri]

            proj_cm = Mesh3D._proj_point_to_plane(tri_verts, cm)

            # update list of top vertices, add edges between vertices as needed
            if not Mesh3D._point_in_tri(tri_verts, proj_cm):
                s1 = Mesh3D._Segment(tri_verts[0], tri_verts[1])
                s2 = Mesh3D._Segment(tri_verts[0], tri_verts[2])
                s3 = Mesh3D._Segment(tri_verts[1], tri_verts[2])

                # TODO: Only using one edge at the moment, should maybe do two
                closest_edges = Mesh3D._closest_segment(proj_cm, [s1, s2, s3])
                closest_edge = closest_edges[0]

                for face in edge_to_faces[closest_edge.tup]:
                    if list(face) != list(tri):
                        topple_face = face
                predecessor = tri_to_vert[tuple(tri)]
                successor = tri_to_vert[tuple(topple_face)]
                predecessor.add_edge(successor)

        prob_map = Mesh3D._compute_prob_map(tri_to_vert.values())

        stable_poses = []
        for face, p in prob_map.items():
            x0 = cvh_verts[face[0]]
            r = cvh_mesh._compute_basis([cvh_verts[i] for i in face])
            if p > 0.0:
                stable_poses.append(sp.StablePose(p, r, x0))
        return stable_poses

    def visualize(self, color=(0.5, 0.5, 0.5), style='surface', opacity=1.0):
        """Plots visualization of mesh using MayaVI.

        Parameters
        ----------
        color : :obj:`tuple` of float
            3-tuple of floats in [0,1] to give the mesh's color

        style : :obj:`str`
            Either 'surface', which produces an opaque surface, or
            'wireframe', which produces a wireframe.

        opacity : float
            A value in [0,1] indicating the opacity of the mesh.
            Zero is transparent, one is opaque.

        Returns
        -------
        :obj:`mayavi.modules.surface.Surface`
            The displayed surface.
        """
        surface = mv.triangular_mesh(self.vertices_[:,0],
                                     self.vertices_[:,1],
                                     self.vertices_[:,2],
                                     self.triangles_, representation=style,
                                     color=color, opacity=opacity)
        return surface

    @staticmethod
    def load(filename, cache_dir,  preproc_script = None):
        """Load a mesh from a file.

        Note
        ----
        If the mesh is not already in .obj format, this requires
        the installation of meshlab. Meshlab has a command called
        meshlabserver that is used to convert the file into a .obj format.

        Parameters
        ----------
        filename : :obj:`str`
            Path to mesh file.
        cache_dir : :obj:`str`
            A directory to store a converted .obj file in, if
            the file isn't already in .obj format.
        preproc_script : :obj:`str`
            The path to an optional script to run before converting
            the mesh file to .obj if necessary.

        Returns
        -------
        :obj:`Mesh3D`
            A 3D mesh object read from the file.
        """
        file_path, file_root = os.path.split(filename)
        file_root, file_ext = os.path.splitext(file_root)
        obj_filename = filename

        if file_ext != Mesh3D.OBJ_EXT:
            obj_filename = os.path.join(cache_dir, file_root + Mesh3D.PROC_TAG + Mesh3D.OBJ_EXT) 
            if preproc_script is None:
                meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(filename, obj_filename)
            else:
                meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\" -s \"%s\"' %(filename, obj_filename, preproc_script) 
            os.system(meshlabserver_cmd)

        if not os.path.exists(obj_filename):
            raise ValueError('Unable to open file %s. It may not exist or meshlab may not be installed.' %(filename))

        # Read mesh from obj file
        return obj_file.ObjFile(obj_filename).read()

    ##################################################################
    # Private Class Methods
    ##################################################################

    def _compute_mass(self):
        """Computes the mesh mass.

        Note
        ----
            Only works for watertight meshes.

        Returns
        -------
        float
            The mass of the mesh.
        """
        return self.density_ * self.get_total_volume()

    def _compute_inertia(self):
        """Computes the mesh inertia matrix.

        Note
        ----
            Only works for watertight meshes.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3x3 inertial matrix.
        """
        C = self.covariance() 
        return self.density_ * (np.trace(C) * np.eye(3) - C)

    def _compute_bb_center(self):
        """Computes the center point of the bounding box.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the center of the bounding box.
        """

        bb_center = (self.min_coords() + self.max_coords()) / 2.0
        return bb_center

    def _compute_com_uniform(self):
        """Computes the center of mass using a uniform mass distribution assumption.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the center of mass.
        """
        total_volume = 0
        weighted_point_sum = np.zeros([1, 3])
        for tri in self.triangles_:
            volume = self._signed_volume_of_tri(tri)
            center = self._center_of_tri(tri)
            weighted_point_sum = weighted_point_sum + volume * center
            total_volume = total_volume + volume
        center_of_mass = weighted_point_sum / total_volume
        return center_of_mass[0]

    def _compute_centroid(self):
        """Computes the centroid (mean) of the mesh's vertices.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            3-ndarray of floats that contains the coordinates
            of the centroid.
        """
        return np.mean(self.vertices_, axis=0)

    def _signed_volume_of_tri(self, tri):
        """Return the signed volume of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a signed volume.

        Returns
        -------
        float
            The signed volume associated with the triangle.
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]

        volume = (1.0 / 6.0) * (v1.dot(np.cross(v2, v3)))
        return volume

    def _center_of_tri(self, tri):
        """Return the center of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a signed volume.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3D point at the center of the triangle
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]
        center = (1.0 / 3.0) * (v1 + v2 + v3)
        return center

    def _covariance_of_tri(self, tri):
        """Return the covariance matrix of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute a covariance.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3x3 covariance matrix of the given triangle.
        """
        v1 = self.vertices_[tri[0], :]
        v2 = self.vertices_[tri[1], :]
        v3 = self.vertices_[tri[2], :]

        A = np.zeros([3,3])
        A[:,0] = v1 - self.center_of_mass_
        A[:,1] = v2 - self.center_of_mass_
        A[:,2] = v3 - self.center_of_mass_
        C = np.linalg.det(A) * A.dot(Mesh3D.C_canonical).dot(A.T)
        return C

    def _area_of_tri(self, tri):
        """Return the area of the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute an area.

        Returns
        -------
        float
            The area of the triangle.
        """
        verts = [self.vertices[i] for i in tri]
        ab = verts[1] - verts[0]
        ac = verts[2] - verts[0]
        return 0.5 * np.linalg.norm(np.cross(ab, ac))

    def _tri_area_percentages(self):
        """Return a list of the percent area each triangle contributes to the
        mesh's surface area.

        Returns
        -------
        :obj:`list` of float
            A list of percentages in [0,1] for each face that represents its
            total contribution to the area of the mesh.
        """
        probs = []
        area = 0.0
        for tri in self.triangles:
            tri_area = self._area_of_tri(tri)
            probs.append(tri_area)
            area += tri_area
        probs = probs / area
        return probs

    def _rand_point_on_tri(self, tri):
        """Return a random point on the given triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle for which we wish to compute an area.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3D point on the triangle.
        """
        verts = [self.vertices[i] for i in tri]
        r1 = np.sqrt(np.random.uniform())
        r2 = np.random.uniform()
        p = (1-r1)*verts[0] + r1*(1-r2)*verts[1] + r1*r2*verts[2]
        return p

    def _compute_proj_area(self, verts):
        """Projects vertices onto the unit sphere from the center of mass
        and computes the projected area.

        Parameters
        ----------
        verts : `list` of `numpy.ndarray` of float
            List of 3-ndarrays of floats that represent the vertices to be
            projected onto the unit sphere.

        Returns
        -------
        float
            The total projected area on the unit sphere.
        """
        cm = self.center_of_mass
        angles = []

        proj_verts = [(v - cm) / np.linalg.norm(v - cm) for v in verts]

        a = math.acos(min(1, max(-1, np.dot(proj_verts[0], proj_verts[1]) /
                                    (np.linalg.norm(proj_verts[0]) * np.linalg.norm(proj_verts[1])))))
        b = math.acos(min(1, max(-1, np.dot(proj_verts[0], proj_verts[2]) /
                                    (np.linalg.norm(proj_verts[0]) * np.linalg.norm(proj_verts[2])))))
        c = math.acos(min(1, max(-1, np.dot(proj_verts[1], proj_verts[2]) /
                                    (np.linalg.norm(proj_verts[1]) * np.linalg.norm(proj_verts[2])))))
        s = (a + b + c) / 2

        try:
            return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*
                                        math.tan((s-b)/2)*math.tan((s-c)/2)))
        except:
            s = s + 0.001
            return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*
                                        math.tan((s-b)/2)*math.tan((s-c)/2)))

    def _compute_basis(self, face_verts):
        """Computes axes for a transformed basis relative to the plane in which input vertices lie.

        Parameters
        ----------
        face_verts : :obj:`numpy.ndarray` of float
            A set of three 3D points that form a plane.

        Returns:
        :obj:`numpy.ndarray` of float
            A 3-by-3 ndarray whose rows are the new basis. This matrix
            can be applied to the mesh to rotate the mesh to lie flat
            on the input face.
        """
        centroid = np.mean(face_verts, axis=0)

        z_o = np.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
        z_o = z_o / np.linalg.norm(z_o)

        # Ensure that all vertices are on the positive halfspace (aka above the table)
        dot_product = (self.vertices - centroid).dot(z_o)
        if dot_product[0] < 0:
            z_o = -z_o

        x_o = np.array([-z_o[1], z_o[0], 0])
        if np.linalg.norm(x_o) == 0.0:
            x_o = np.array([1, 0, 0])
        else:
            x_o = x_o / np.linalg.norm(x_o)
        y_o = np.cross(z_o, x_o)
        y_o = y_o / np.linalg.norm(y_o)

        R = np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])

        # rotate the vertices and then align along the principal axes
        rotated_vertices = R.dot(self.vertices.T)
        xy_components = rotated_vertices[:2,:].T

        pca = sklearn.decomposition.PCA(n_components = 2)
        pca.fit(xy_components)
        comp_array = pca.components_
        x_o = R.T.dot(np.array([comp_array[0,0], comp_array[0,1], 0]))
        y_o = np.cross(z_o, x_o)
        return np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])

    class _Segment:
        """Object representation of a finite line segment in 3D space.

        Attributes
        ----------
        p1 : :obj:`numpy.ndarray` of float
            The first endpoint of the line segment
        p2 : :obj:`numpy.ndarray` of float
            The second endpoint of the line segment
        tup : :obj:`tuple` of :obj:`tuple` of float
            A tuple representation of the segment, with the two
            endpoints arranged in lexicographical order.
        """

        def __init__(self, p1, p2):
            """Creates a Segment with given endpoints.

            Parameters
            ----------
            p1 : :obj:`numpy.ndarray` of float
                The first endpoint of the line segment
            p2 : :obj:`numpy.ndarray` of float
                The second endpoint of the line segment
            """
            self.p1 = p1
            self.p2 = p2
            self.tup = self._ordered_tuple()

        def dist_to_point(self, point):
            """Computes the distance from the segment to the given 3D point.

            Parameters
            ----------
            point : :obj:`numpy.ndarray` of float
                The 3D point to measure distance to.

            Returns
            -------
            float
                The euclidean distance between the segment and the point.
            """
            p1, p2 = self.p1, self.p2
            ap = point - p1
            ab = p2 - p1
            proj_point = p1 + (np.dot(ap, ab) / np.dot(ab, ab)) * ab
            if self._contains_proj_point(proj_point):
                return np.linalg.norm(point - proj_point)
            else:
                return min(np.linalg.norm(point - p1),
                        np.linalg.norm(point - p2))

        def _contains_proj_point(self, point):
            """Is the given 3D point (assumed to be on the line that contains
            the segment) actually within the segment?

            Parameters
            ----------
            point : :obj:`numpy.ndarray` of float
                The 3D point to check against.

            Returns
            -------
            bool
                True if the point was within the segment or False otherwise.
            """
            p1, p2 = self.p1, self.p2
            return (point[0] >= min(p1[0],p2[0]) and point[0] <= max(p1[0],p2[0]) and
                    point[1] >= min(p1[1],p2[1]) and point[1] <= max(p1[1],p2[1]) and
                    point[2] >= min(p1[2],p2[2]) and point[2] <= max(p1[2],p2[2]))

        def _ordered_tuple(self):
            """Returns an ordered tuple that represents the segment.

            The points within are ordered lexicographically.

            Returns
            -------

            tup : :obj:`tuple` of :obj:`tuple` of float
                A tuple representation of the segment, with the two
                endpoints arranged in lexicographical order.
            """
            if (self.p1.tolist() > self.p2.tolist()):
                return (tuple(self.p1), tuple(self.p2))
            else:
                return (tuple(self.p2), tuple(self.p1))

    class _GraphVertex:
        """A directed graph vertex that links a probability to a face.
        """

        def __init__(self, probability, face):
            """Create a graph vertex with given probability and face.

            Parameters
            ----------
            probability : float
                Probability associated with this vertex.
            face : :obj:`numpy.ndarray` of int
                A 3x3 array that represents the face
                associated with this vertex. Each row is a list
                of vertices in one face.
            """
            self.probability = probability
            self.children = []
            self.parents = []
            self.face = face
            self.is_sink = True if not self.children else False
            self.has_parent = False
            self.num_parents = 0

        def add_edge(self, child):
            """Connects this vertex to the input child vertex.

            Parameters
            ----------
            child : :obj:`_GraphVertex`
                The child to link to.
            """
            self.is_sink = False
            self.children.append(child)
            child.parents.append(self)
            child.has_parent = True
            child.num_parents += 1

    @staticmethod
    def _max_edge_length(tri, vertices):
        """Compute the maximum edge length of a triangle.

        Parameters
        ----------
        tri : :obj:`numpy.ndarray` of int
            The triangle of interest.

        vertices : :obj:`numpy.ndarray` of `numpy.ndarray` of float
            The set of vertices which the face references.

        Returns
        -------
        float
            The max edge length of the triangle.
        """
        v0 = np.array(vertices[tri[0]])
        v1 = np.array(vertices[tri[1]])
        v2 = np.array(vertices[tri[2]])
        max_edge_len = max(np.linalg.norm(v1-v0),
                       max(np.linalg.norm(v1-v0),
                           np.linalg.norm(v2-v1)))
        return max_edge_len

    @staticmethod
    def _proj_point_to_plane(tri_verts, point):
        """Project the given point onto the plane containing the three points in
        tri_verts.

        Parameters
        ----------
        tri_verts : :obj:`numpy.ndarray` of float
            A list of three 3D points that defines a plane.
        point : :obj:`numpy.ndarray` of float
            The 3D point to project onto the plane.
        """

        # Compute a normal vector to the triangle
        v0 = tri_verts[2] - tri_verts[0]
        v1 = tri_verts[1] - tri_verts[0]
        n = np.cross(v0, v1)
        n = n / np.linalg.norm(n)

        # Compute distance from the point to the triangle's plane
        # by projecting a vector from the plane to the point onto
        # the normal vector
        dist = np.dot(n, point - tri_verts[0])

        # Project the point back along the normal vector
        return (point - dist*n)

    @staticmethod
    def _point_in_tri(tri_verts, point):
        """Is the given point contained in the given triangle?

        Parameters
        ----------
        tri_verts : :obj:`list` of :obj:`numpy.ndarray` of float
            A list of three 3D points that definie a triangle.

        point : :obj:`numpy.ndarray` of float
            A 3D point that should be coplanar with the triangle.

        Returns
        -------
        bool
            True if the point is in the triangle, False otherwise.
        """
        # Implementation provided by http://blackpawn.com/texts/pointinpoly/

        # Compute vectors
        v0 = tri_verts[2] - tri_verts[0]
        v1 = tri_verts[1] - tri_verts[0]
        v2 = point - tri_verts[0]

        # Compute Dot Products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute Barycentric Coords
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0.0 and v >= 0.0 and u + v <= 1.0)

    @staticmethod
    def _closest_segment(point, line_segments):
        """Returns the finite line segment(s) the least distance from the input point.

        Parameters
        ----------
        point : :obj:`numpy.ndarray` of float
            The 3D point to measure distance to.
        line_segments: :obj:`list` of :obj:`_Segments`
            The list of line segments.

        Returns
        -------
        :obj:`list` of :obj:`_Segments`
            The list of line segments that were closest to the input point.
        """
        min_dist = sys.maxint
        min_segs = []
        distances = []
        segments = []
        common_endpoint = None

        for segment in line_segments:
            dist = segment.dist_to_point(point)
            distances.append(dist)
            segments.append(segment)
            if dist < min_dist:
                min_dist = dist

        for i in range(len(distances)):
            if min_dist + 0.000001 >= distances[i]:
                min_segs.append(segments[i])

        return min_segs

    @staticmethod
    def _compute_prob_map(vertices):
        """Creates a map from faces to static stability probabilities.

        Parameters
        ----------
        vertices : :obj:`list` of :obj:`_GraphVertex`

        Returns
        -------
        :obj:`dictionary` of :obj:`tuple` of int to float
            Maps tuple representations of faces to probabilities.
        """
        prob_mapping = {}
        for vertex in vertices:
            c = vertex
            visited = []
            while not c.is_sink:
                if c in visited:
                    break
                visited.append(c)
                c = c.children[0]

            if tuple(c.face) not in prob_mapping.keys():
                prob_mapping[tuple(c.face)] = 0.0
            prob_mapping[tuple(c.face)] += vertex.probability

        for vertex in vertices:
            if not vertex.is_sink:
                prob_mapping[tuple(vertex.face)] = 0
        return prob_mapping

if __name__ == '__main__':
    pass
