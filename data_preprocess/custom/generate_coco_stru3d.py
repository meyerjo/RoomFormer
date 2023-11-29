import argparse
import json
import os
import sys
from tqdm import tqdm
from stru3d_utils import generate_density, normalize_annotations, parse_floor_plan_polys, generate_coco_dict

sys.path.append('../.')
from common_utils import read_scene_pc, export_density
from pathlib import Path


### Note: Some scenes have missing/wrong annotations. These are the indices that you should additionally exclude 
### to be consistent with MonteFloor and HEAT:

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_stru3d', type=str, help='path to output folder')
    
    args = a.parse_args()
    return args

def main(args):
    data_root = args.data_root

    data_parts = Path(data_root).rglob("*.ply")

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    for i, ply_file in enumerate(data_parts):
        points = read_scene_pc(ply_file)
        xyz = points[:, :3]
        density, normalization_dict = generate_density(xyz, width=256, height=256)
        export_density(density, outFolder, scene_id=i)


if __name__ == "__main__":

    main(config())