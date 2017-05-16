try:
    import meshrender
except:
    print 'Unable to import meshrender shared library! Rendering will not work. Likely due to missing Boost.Numpy'
    print 'Boost.Numpy can be installed following the instructions in https://github.com/ndarray/Boost.NumPy'
from mesh import Mesh3D
from image_converter import ImageToMeshConverter
from obj_file import ObjFile
from off_file import OffFile
from sdf import Sdf, Sdf3D
from sdf_file import SdfFile
from stable_pose import StablePose
from stp_file import StablePoseFile
from urdf_writer import UrdfWriter

try:
    from mesh_renderer import ViewsphereDiscretizer, PlanarWorksurfaceDiscretizer, VirtualCamera, SceneObject
    from random_variables import CameraSample, RenderSample, UniformViewsphereRandomVariable, UniformPlanarWorksurfaceRandomVariable, UniformPlanarWorksurfaceImageRandomVariable
except:
    print 'Unable to import mesh rendering! Likely due to missing Boost.Numpy'
    print 'Boost.Numpy can be installed following the instructions in https://github.com/ndarray/Boost.NumPy'

__all__ = ['Mesh3D',
           'ViewsphereDiscretizer', 'PlanarWorksurfaceDiscretizer', 'VirtualCamera', 'SceneObject',
           'ImageToMeshConverter',
           'ObjFile', 'OffFile',
           'Sdf', 'Sdf3D',
           'SdfFile',
           'StablePose',
           'StablePoseFile',
           'CameraSample',
           'RenderSample',
           'UniformViewsphereRandomVariable',
           'UniformPlanarWorksurfaceRandomVariable',
           'UniformPlanarWorksurfaceImageRandomVariable'
           'UrdfWriter'
       ]
