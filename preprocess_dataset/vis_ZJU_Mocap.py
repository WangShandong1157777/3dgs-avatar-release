"""
=============================
@author: Shandong
@email: shandong.wang@intel.com
@time: 2024/5/30:下午4:42
@IDE: PyCharm
=============================
"""

import os
import glob
import cv2
import numpy as np
from smplx import SMPL
import torch
import pyrender
import trimesh
import json

def print_hi():
    print(f'Hi, this is to show smpl mesh onto original images')

def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img

if __name__ == "__main__":
    print_hi()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    args = parser.parse_args()
    args.path = "./data/ZJUMoCap/preprocessed_for_3DGS"
    subject = "CoreView_377"
    cam_id = "1"

    poses_path = os.path.join(args.path, subject, "models")

    smpl_dir = "./body_models/smpl"
    smpl = SMPL(model_path=smpl_dir, gender='NEUTRAL', batch_size=1)

    camera_path = os.path.join(args.path, subject, "cam_params.json")
    with open(os.path.join(camera_path), 'r') as f:
        cameras_dict = json.load(f)
    cam_K = np.array(cameras_dict[cam_id]['K'])
    cam_R = np.array(cameras_dict[cam_id]['R'])
    cam_T = np.array(cameras_dict[cam_id]['T'])
    cam_param = {}
    cam_param['focal'] = np.array((cam_K[0,0], cam_K[1,1]))
    cam_param['princpt'] = np.array((cam_K[0,2], cam_K[1,2]))

    imgs_path = os.path.join(args.path, subject, cam_id, "*.jpg")
    msks_path = os.path.join(args.path, subject, cam_id, "*.png")
    imgs = sorted(glob.glob(imgs_path))
    msks = sorted(glob.glob(msks_path))

    cv2.namedWindow('img', 0)
    cv2.namedWindow('smpl', 0)
    cv2.namedWindow('smpl2', 0)

    i = 0
    for img, mask in zip(imgs, msks):
        # show image with mask
        img = cv2.imread(img)
        mask = cv2.imread(mask)
        img[mask == 0] = 0
        cv2.imshow("img", img)

        ### show smpl mesh
        pose_path = poses_path + f"/{i:06d}.npz"
        model_dict = np.load(pose_path)
        smpl_param = {}
        smpl_param['betas'] = model_dict['betas'].astype(np.float32)
        smpl_param['global_orient'] = model_dict['root_orient'].astype(np.float32)
        smpl_param['pose_body'] = model_dict['pose_body'].astype(np.float32)
        smpl_param['pose_hand'] = model_dict['pose_hand'].astype(np.float32)
        smpl_param['trans'] = model_dict['trans'].astype(np.float32)
        smpl_param["body_pose"] = np.concatenate((smpl_param['pose_body'], smpl_param['pose_hand']), axis=0)
        # pyrender vis
        output = smpl.forward(
            betas = torch.from_numpy(smpl_param['betas'].reshape(1,10)).float(),
            global_orient=torch.from_numpy(smpl_param["global_orient"].reshape(1,3)).float(),
            body_pose=torch.from_numpy(smpl_param["body_pose"].reshape(1,69)).float(),
            transl=torch.from_numpy(smpl_param["trans"].reshape(1,3)).float(),
        )
        joints = output.joints.detach().numpy()[0]  # (45,3)
        pelvis = joints[0]
        vertices = output.vertices.detach().numpy()[0]  # (6890,3)
        # by cam_extrinsic
        vertices = np.dot(vertices, cam_R.T)
        vertices = vertices + cam_T.T
        rendered_img = render_mesh(img, vertices, smpl.faces, cam_param)
        cv2.imshow('smpl', rendered_img.astype(np.uint8))

        # smpl params in camera space
        pelvis = joints[0]
        pelvis_new = np.dot(pelvis, cam_R.T) + cam_T.T
        global_trans = pelvis_new - (pelvis-smpl_param["trans"].reshape(1,3))  # new one
        global_ori = smpl_param["global_orient"].reshape(1,3)
        rotmat, _ = cv2.Rodrigues(global_ori)
        rotmat = np.dot(cam_R, rotmat)
        global_ori, _ = cv2.Rodrigues(rotmat)
        # smpl_poses[f, 0:3] = global_ori.T  # new one
        vertices = smpl.forward(
            global_orient=torch.from_numpy(global_ori.reshape(1,3)).float(),
            body_pose=torch.from_numpy(smpl_param["body_pose"].reshape(1,69)).float(),
            transl=torch.from_numpy(global_trans).float(),
        ).vertices.detach().numpy()[0]  # each frame
        rendered_img = render_mesh(img, vertices, smpl.faces, cam_param)
        # cv2.imwrite('smpl2.jpg', rendered_img)
        cv2.imshow('smpl2', rendered_img.astype(np.uint8))

        cv2.waitKey(10)

        i += 1

