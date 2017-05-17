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
from meshpy import MaterialProperties, LightingProperties, ObjFile, VirtualCamera, ViewsphereDiscretizer, SceneObject

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
    stable_pose = mesh.stable_poses()[0]
    T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')
    #T_obj_world = RigidTransform(from_frame='obj',
    #                             to_frame='world')

    # load camera intrinsics
    camera_intr = CameraIntrinsics.load('data/camera_intr/primesense_carmine_108.intr')
    
    # create virtual camera
    virtual_camera = VirtualCamera(camera_intr)


    # create lighting props
    T_light_camera = RigidTransform(translation=[0,-0.5,0],
                                    from_frame='light',
                                    to_frame=camera_intr.frame)
    light_props = LightingProperties(T_light_camera=T_light_camera)

    # create material props
    mat_props = MaterialProperties(color=np.array([244,164,96]))

    # create scene objects
    table_mesh = ObjFile('data/meshes/table.obj').read()
    scene_objs = {'table': SceneObject(table_mesh, T_obj_world.inverse(),
                                       mat_props=MaterialProperties(color=np.array([0,255,0])))}
    for name, scene_obj in scene_objs.iteritems():
        virtual_camera.add_to_scene(name, scene_obj)

    # camera pose
    cam_dist = 0.25
    T_camera_world = RigidTransform(rotation=np.array([[0, 1, 0],
                                                       [1, 0, 0],
                                                       [0, 0, -1]]),
                                    translation=[0,0,0.5],
                                    from_frame=camera_intr.frame,
                                    to_frame='world')
    T_obj_camera = T_camera_world.inverse() * T_obj_world

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
                                            mat_props=mat_props,
                                            light_props=light_props,
                                            debug=True)
    vis.figure()
    vis.imshow(renders[0].image)
    vis.show()
