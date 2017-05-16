"""
Renders an image for a mesh in each stable pose to demo the rendering interface.
Author: Jeff Mahler
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import sys

from core import RigidTransform
from perception import CameraIntrinsics, ObjectRender, RenderMode
from meshpy import ObjFile, VirtualCamera, ViewsphereDiscretizer

from visualization import Visualizer2D as vis
from visualization import Visualizer3D as vis3d

if __name__ == '__main__':
    # parse args
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_filename', type=str, help='filename for .OBJ mesh file to render')
    args = parser.parse_args()

    # read data
    mesh_filename = args.mesh_filename
    _, mesh_ext = os.path.splitext(mesh_filename)
    if mesh_ext != '.obj':
        raise ValueError('Must provide mesh in Wavefront .OBJ format!') 
    mesh = ObjFile(mesh_filename).read()

    # load camera intrinsics
    camera_intr = CameraIntrinsics.load('data/camera_intr/primesense_carmine_108.intr')
    
    # create virtual camera
    virtual_camera = VirtualCamera(camera_intr)

    # camera pose
    cam_dist = 0.25
    T_obj_camera = RigidTransform(translation=[0,0,0.5],
                                  from_frame='obj',
                                  to_frame=camera_intr.frame)

    # show mesh
    if False:
        vis3d.figure()
        vis3d.mesh(mesh, T_obj_camera)
        vis3d.pose(RigidTransform(), alpha=0.1)
        vis3d.pose(T_obj_camera, alpha=0.1)
        vis3d.show()

    # render depth image
    renders = virtual_camera.wrapped_images(mesh,
                                            [T_obj_camera],
                                            RenderMode.COLOR,
                                            debug=True)
    vis.figure()
    vis.imshow(renders[0].image)
    vis.show()
