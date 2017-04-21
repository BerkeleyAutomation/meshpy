"""
Regressive test for stable poses. Qualitative only.
Author: Jeff Mahler
"""
import IPython
import numpy as np
import os
import sys

from core import RigidTransform
from meshpy import ObjFile, Mesh3D
from visualization import Visualizer3D as vis

if __name__ == '__main__':
    mesh_name = sys.argv[1]

    # read mesh
    mesh = ObjFile(mesh_name).read()

    T_obj_table = RigidTransform(rotation=RigidTransform.random_rotation(),
                                 from_frame='obj', to_frame='table')
    stable_pose = mesh.resting_face(T_obj_table)


    table_dim = 0.3
    T_obj_table_plot = mesh.get_T_surface_obj(T_obj_table)
    T_obj_table_plot.translation[0] += 0.1
    vis.figure()
    vis.mesh(mesh, T_obj_table_plot, 
             color=(1,0,0), style='surface')
    vis.mesh_stable_pose(mesh, stable_pose, dim=table_dim,
                         color=(0,1,0), style='surface')
    vis.show()
    exit(0)

    # compute stable poses
    vis.figure()
    vis.mesh(mesh, color=(1,1,0), style='surface')
    vis.mesh(mesh.convex_hull(), color=(1,0,0))

    stable_poses = mesh.stable_poses()
    
    vis.show()
