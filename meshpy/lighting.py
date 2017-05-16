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
                 ambient_reflectance=np.array([0.2,0.2,0.2,1]),
                 diffuse_reflectance=np.array([0.8,0.8,0.8,1]),
                 specular_reflectance=np.array([0,0,0,1]),
                 shininess=0):
        # set params
        self.color = color
        self.ambient_reflectance = ambient_reflectance
        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.shininess = shininess
                 

    @property
    def arr(self):
        """ Returns the material properties as a contiguous numpy array. """
        return np.r_[self.color,
                     self.ambient_reflectance,
                     self.diffuse_reflectance,
                     self.specular_reflectance,
                     self.shininess].astype(np.float64)

class LightingProperties(object):
    """ Struct to encapsulate lighting properties for
    OpenGL rendering.
    """
    def __init__(self, ambient_intensity=np.array([0,0,0,1]),
                 diffuse_intensity=np.array([1,1,1,1]),
                 specular_intensity=np.array([1,1,1,1]),
                 T_light_world=RigidTransform(rotation=np.array([[0, 1, 0],
                                                                 [1, 0, 0],
                                                                 [0, 0, -1]]),
                                              translation=[0,0,-0.1],
                                              from_frame='light',
                                              to_frame='world'),
                 spot_cutoff=180.0):
        self.ambient_intensity = ambient_intensity
        self.diffuse_intensity = diffuse_intensity
        self.specular_intensity = specular_intensity
        self.T_light_world = T_light_world
        self.spot_cutoff = spot_cutoff

    @property
    def arr(self):
        """ Returns the lighting properties as a contiguous numpy array. """
        return np.r_[self.ambient_intensity,
                     self.diffuse_intensity,
                     self.specular_intensity,
                     self.T_light_world.translation,
                     self.T_light_world.z_axis,                     
                     self.spot_cutoff].astype(np.float64)

