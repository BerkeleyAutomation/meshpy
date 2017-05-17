"""
Classes for lighting in renderer
Author: Jeff Mahler
"""
import numpy as np

from core import RigidTransform

class Color(object):
    WHITE = np.array([255, 255, 255])
    BLACK = np.array([0, 0, 0])
    RED   = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE  = np.array([0, 0, 255])

class MaterialProperties(object):
    """ Struct to encapsulate material properties for
    OpenGL rendering.

    Attributes
    ----------
    color : :obj:`numpy.ndarray`
        3-array of integers between 0 and 255
    """
    def __init__(self, color=Color.WHITE,
                 ambient=0.2,
                 diffuse=0.8,
                 specular=0,
                 shininess=0):
        # set params
        self.color = np.array(color)
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
                 

    @property
    def arr(self):
        """ Returns the material properties as a contiguous numpy array. """
        return np.r_[self.color,
                     self.ambient * np.ones(3), 1,
                     self.diffuse * np.ones(3), 1,
                     self.specular * np.ones(3), 1,
                     self.shininess].astype(np.float64)

class LightingProperties(object):
    """ Struct to encapsulate lighting properties for
    OpenGL rendering.
    """
    def __init__(self, ambient=0,
                 diffuse=1,
                 specular=1,
                 T_light_camera=RigidTransform(rotation=np.eye(3),
                                               translation=np.zeros(3),
                                               from_frame='light',
                                               to_frame='camera'),
                 cutoff=180.0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.T_light_camera = T_light_camera
        self.cutoff = cutoff
        self.T_light_obj = None

    def set_pose(self, T_obj_camera):
        self.T_light_obj = T_obj_camera.inverse() * self.T_light_camera.as_frames('light', T_obj_camera.to_frame)

    @property
    def arr(self):
        """ Returns the lighting properties as a contiguous numpy array. """
        if self.T_light_obj is None:
            raise ValueError('Need to set pose relative to object!')
        return np.r_[self.ambient * np.ones(3), 1,
                     self.diffuse * np.ones(3), 1,
                     self.specular * np.ones(3), 1,
                     self.T_light_obj.translation,
                     self.T_light_obj.z_axis,                     
                     self.cutoff].astype(np.float64)

