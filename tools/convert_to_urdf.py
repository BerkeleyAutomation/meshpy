"""
Utility script to convert obj files to urdfs
"""
import os, argparse, logging
from os.path import isfile, join, splitext, basename, exists
from meshpy import UrdfWriter, ObjFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_directory')
    parser.add_argument('target_directory')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Converting objs from {} to {}".format(args.source_directory,
            args.target_directory))

    for f in os.listdir(args.source_directory):
        src_path = join(args.source_directory, f)
        if isfile(src_path) and src_path.endswith(".obj"):
            name = splitext(basename(src_path))[0]
            tar_path = join(args.target_directory, name)
            if not exists(tar_path):
                os.makedirs(tar_path)
            logging.info("Writing {}".format(src_path))
            obj = ObjFile(src_path)
            urdf = UrdfWriter(tar_path)
            mesh = obj.read()
            urdf.write(mesh)
