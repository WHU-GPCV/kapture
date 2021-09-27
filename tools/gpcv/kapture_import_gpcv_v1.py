#!/usr/bin/env python3
# Copyright 2021-present GPCV. Under BSD 3-clause license

"""
Script to import the special model of gpcv into a kapture.
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from typing import Optional
from PIL import Image
from tqdm import tqdm
# kapture
# import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.paths import path_secure

logger = logging.getLogger('gpcv')

def get_camera_param(type:str, line:str):

    if type == "PINHOLE":
        params_item = line.split(' ')
        cam_id = params_item[0]
        width = int(params_item[1])
        height = int(params_item[2])
        pixel_size = float(params_item[3])
        fx = float(params_item[4])
        fy = float(params_item[5])
        cx = float(params_item[6])
        cy = float(params_item[7])
        K1 = float(params_item[8])
        K2 = float(params_item[9])
        K3 = float(params_item[10])
        P1 = float(params_item[11])
        P2 = float(params_item[12])

        return [cam_id, kapture.CameraType.PINHOLE, [width, height, fx, fy, cx, cy]]
    
    
def load_sensor_and_rigid(path: str):
    sensors = kapture.Sensors()
    rigs = kapture.Rigs()

    f = open(path)

    num_line = f.readline().strip()
    num_line_item = num_line.split(' ')
    num = int(num_line_item[-1])

    camera_type_line = f.readline().strip()
    camera_type_item = camera_type_line.split(' ')
    camera_type = camera_type_item[-1]

    num_line = f.readline().strip()
    num_line = f.readline().strip()

    for index in range(num):
        line = f.readline().strip()        
        [cam_id, cam_type, cam_params] = get_camera_param(camera_type, line)

        sensors[cam_id] = kapture.Camera(
            name=cam_id,
            camera_type=cam_type,
            camera_params=cam_params
        )

        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        T = np.array([0, 0, 0])
        Rt = np.vstack((np.hstack((R, T.reshape(3, 1))), np.array([0, 0, 0, 1])))
        rigs["body", cam_id] = kapture.PoseTransform(quaternion.from_rotation_matrix(R), T)
        
    return [sensors, rigs]


def load_image_path(path: str):
    result = {}
    f = open(path)

    num_line = f.readline().strip()
    num = int(num_line)
    for index in range(num):
        line = f.readline().strip()
        if line[0] != '#':
            pass
        
        items = line.split(' ')
        img_id = int(items[0])
        result[img_id] = items[1]

    return result


def load_image_info(path: str):
    result = {}
    f = open(path)

    num_line = f.readline().strip()
    num_line_item = num_line.split(' ')
    num = int(num_line_item[-1])

    f.readline().strip()
    f.readline().strip()

    for index in range(num):
        line = f.readline().strip()
        params_item = line.split(' ')

        img_id = int(params_item[0])
        cam_id = params_item[1]
        R1 = float(params_item[2])
        R2 = float(params_item[3])
        R3 = float(params_item[4])
        R4 = float(params_item[5])
        R5 = float(params_item[6])
        R6 = float(params_item[7])
        R7 = float(params_item[8])
        R8 = float(params_item[9])
        R9 = float(params_item[10])
        t1 = float(params_item[11])
        t2 = float(params_item[12])
        t3 = float(params_item[13])

        result[img_id] = [cam_id, R1, R2, R3, R4, R5, R6, R7, R8, R9, t1, t2, t3]

    return result


def load_image_and_trajection(image_info_path: str, image_path_path: str):

    snapshots = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()

    image_info = load_image_info(image_info_path)
    image_path = load_image_path(image_path_path)

    for image_id in image_info:
        rotation_mat = np.array([[(image_info[image_id])[1], (image_info[image_id])[2], (image_info[image_id])[3]],
                                 [(image_info[image_id])[4], (image_info[image_id])[5], (image_info[image_id])[6]],
                                 [(image_info[image_id])[7], (image_info[image_id])[8], (image_info[image_id])[9]]])
        position_vec =  np.array([[(image_info[image_id])[10]],
                                  [(image_info[image_id])[11]],
                                  [(image_info[image_id])[12]]])
        rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
        pose_world_from_cam = kapture.PoseTransform(r=rotation_quat, t=position_vec)
        pose_cam_from_world = pose_world_from_cam.inverse()
        trajectories[image_id, (image_info[image_id])[0]] = pose_cam_from_world

        image_absolute_path = image_path[image_id]
        snapshots[image_id, (image_info[image_id])[0]] = image_absolute_path

    return [snapshots, trajectories]


def import_gpcv(gpcv_path: str,
                kapture_dir_path: str,
                force_overwrite_existing: bool = False,
                images_import_method: TransferAction = TransferAction.skip,
                partition: Optional[str] = None
                ) -> None:
    """
    Imports gpcv dataset and save them as kapture.

    :param gpcv_path: path to the gpcv sequence root path
    :param kapture_dir_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param partition: if specified = 'mapping' or 'query'. Requires gpcv_path/TestSplit.txt or TrainSplit.txt
                    to exists.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    logger.info('loading all content ...')

    # images and poses
    logger.info('populating image files and pose ...')
    [snapshots, trajectories] = load_image_and_trajection(gpcv_path + "/image_info.txt", 
                                                          gpcv_path + "/image_path.txt")

    # sensors and rigid
    logger.info('populating sensor files and rig ...')
    [sensors, rigs] = load_sensor_and_rigid(gpcv_path + "/camera_info.txt")

    # import (copy) image files.
    logger.info('copying image files ...')
    image_filenames = [f for _, _, f in kapture.flatten(snapshots)]
    import_record_data_from_dir_auto(gpcv_path, kapture_dir_path, image_filenames, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(
        records_camera=snapshots,
        rigs=rigs,
        trajectories=trajectories,
        sensors=sensors)

    logger.info('writing imported data ...')
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_gpcv_command_line() -> None:
    """
    Imports GPCV Dataset and save them as kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='Imports GPCV Dataset files to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input', required=True,
                        help='input path Dataset sequence root path')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    parser.add_argument('-p', '--partition', default=None, choices=['mapping', 'query'],
                        help='limit to mapping or query sequences only (using authors split files).')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_gpcv(gpcv_path=args.input,
                kapture_dir_path=args.output,
                force_overwrite_existing=args.force,
                images_import_method=args.image_transfer,
                partition=args.partition)


if __name__ == '__main__':
    import_gpcv_command_line()
