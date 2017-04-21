"""
Regressive test for stable poses. Qualitative only.
Author: Jeff Mahler
"""
import IPython
import numpy as np
import os
import sys

from meshpy import ObjFile, Mesh3D
from visualization import Visualizer3D as vis

if __name__ == '__main__':
    mesh_name = sys.argv[1]

    # read mesh
    mesh = ObjFile(mesh_name).read()

    # compute stable poses
    vis.figure()
    vis.mesh(mesh, color=(1,1,0), style='surface')
    vis.mesh(mesh.convex_hull(), color=(1,0,0))

    stable_poses = mesh.stable_poses()
    
    vis.show()
